import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder
import gc
import warnings
warnings.filterwarnings('ignore')

TRANSACTIONS_FILE = 'train_transactions.csv'
USERS_FILE = 'train_users.csv'

def build_ml_ready_dataset():
    print("🚀 СТАРТ: Завантаження даних...")
    df_trans = pd.read_csv(TRANSACTIONS_FILE)
    df_users = pd.read_csv(USERS_FILE)

    print("\n⏱️ БЛОК 1: Обробка часу та нові фічі...")
    # Перетворюємо текст на об'єкти часу
    df_users['timestamp_reg'] = pd.to_datetime(df_users['timestamp_reg'], errors='coerce')
    df_trans['timestamp_tr'] = pd.to_datetime(df_trans['timestamp_tr'], errors='coerce')

    # Підтягуємо країну та час реєстрації до транзакцій
    df_trans = df_trans.merge(df_users[['id_user', 'reg_country', 'timestamp_reg']], on='id_user', how='left')
    
    # Створюємо логічні прапорці (1 - так, 0 - ні)
    df_trans['is_cross_border_payment'] = (df_trans['card_country'] != df_trans['payment_country']).astype(int)
    df_trans['is_reg_mismatch'] = (df_trans['reg_country'] != df_trans['payment_country']).astype(int)
    df_trans['is_fail'] = (df_trans['status'] == 'fail').astype(int)
    
    # Шукаємо конкретні помилки
    df_trans['is_3ds_error'] = (df_trans['error_group'] == '3ds error').astype(int)
    df_trans['is_fraud_error'] = (df_trans['error_group'] == 'fraud').astype(int)

    # Вираховуємо час від реєстрації до першої транзакції (у хвилинах)
    first_trans = df_trans.groupby('id_user')['timestamp_tr'].min().reset_index()
    first_trans = first_trans.merge(df_users[['id_user', 'timestamp_reg']], on='id_user')
    first_trans['time_to_first_tx_mins'] = (first_trans['timestamp_tr'] - first_trans['timestamp_reg']).dt.total_seconds() / 60.0
    first_trans['time_to_first_tx_mins'] = first_trans['time_to_first_tx_mins'].clip(lower=0)

    # Ідея 1: Маркери довіри
    df_trans['is_manual_entry'] = (df_trans['transaction_type'] == 'card_init').astype(int)
    df_trans['is_safe_pay'] = (df_trans['transaction_type'].isin(['apple-pay', 'google-pay'])).astype(int)
    
    # Ідея 3: Відхилені суми
    df_trans['failed_amount'] = np.where(df_trans['status'] == 'fail', df_trans['amount'], 0)

    print("\n📊 БЛОК 2: Агрегація досьє користувача...")
    user_agg = df_trans.groupby('id_user').agg(
        total_tx=('id_user', 'count'),
        fail_count=('is_fail', 'sum'),
        cross_border_payment=('is_cross_border_payment', 'sum'),
        reg_mismatch=('is_reg_mismatch', 'sum'),
        total_amount=('amount', 'sum'),
        avg_amount=('amount', 'mean'),
        max_amount=('amount', 'max'),
        unique_cards=('card_mask_hash', 'nunique'),
        unique_card_holders=('card_holder', 'nunique'),     # Скільки різних імен вводив
        unique_devices=('transaction_type', 'nunique'),     # Чи міняв способи оплати
        error_3ds_count=('is_3ds_error', 'sum'),
        error_fraud_count=('is_fraud_error', 'sum'),        # <--- ОСЬ ЦЯ КОМА БУЛА ПРОПУЩЕНА

        manual_entry_count=('is_manual_entry', 'sum'),
        safe_pay_count=('is_safe_pay', 'sum'),
        failed_amount_total=('failed_amount', 'sum'),
        last_tx_time=('timestamp_tr', 'max'), # Для швидкості паніки
        first_tx_time=('timestamp_tr', 'min') # Для швидкості паніки
    ).reset_index()

    # Ідея 2: Математика швидкості (Velocity)
    # Скільки хвилин юзер був активний (від першої до останньої транзакції)
    user_agg['active_minutes'] = (user_agg['last_tx_time'] - user_agg['first_tx_time']).dt.total_seconds() / 60.0
    # Захист від ділення на нуль (якщо всього 1 транзакція)
    user_agg['active_minutes'] = user_agg['active_minutes'].replace(0, 0.1) 
    # Транзакцій за хвилину
    user_agg['tx_per_minute'] = user_agg['total_tx'] / user_agg['active_minutes']
    
    # Видаляємо сирі дати, вони моделі не потрібні
    user_agg = user_agg.drop(columns=['last_tx_time', 'first_tx_time'])

    user_agg['fail_rate'] = user_agg['fail_count'] / user_agg['total_tx']
    user_agg = user_agg.merge(first_trans[['id_user', 'time_to_first_tx_mins']], on='id_user', how='left')

    print("\n🕸️ БЛОК 3: Швидкі Графи (Топологія)...")
    edges_df = df_trans[['id_user', 'card_mask_hash']].drop_duplicates().dropna()
    edges_df['u_node'] = 'U_' + edges_df['id_user'].astype(str)
    edges_df['c_node'] = 'C_' + edges_df['card_mask_hash'].astype(str)

    G = nx.from_pandas_edgelist(edges_df, 'u_node', 'c_node')
    
    pools = list(nx.connected_components(G))
    pool_features = []
    for comp in pools:
        pool_size = len(comp)
        for node in comp:
            if str(node).startswith('U_'):
                pool_features.append({
                    'id_user': int(str(node).replace('U_', '')), 
                    'graph_pool_size': pool_size,
                    'graph_user_degree': G.degree(node)
                })
    df_graph = pd.DataFrame(pool_features)
    pr = nx.pagerank(G, alpha=0.85)
    df_graph['graph_pagerank'] = df_graph['id_user'].apply(lambda x: pr.get(f'U_{x}', 0))

    del G, edges_df
    gc.collect()

    print("\n🧩 БЛОК 4: Злиття та кодування...")
    final_df = df_users.merge(user_agg, on='id_user', how='left')
    final_df = final_df.merge(df_graph, on='id_user', how='left')
    final_df = final_df.fillna(0) # Якщо не було транзакцій, ставимо 0

    le = LabelEncoder()
    cat_cols = ['gender', 'reg_country', 'traffic_type']
    for col in cat_cols:
        final_df[col] = le.fit_transform(final_df[col].astype(str))

    cols_to_drop = ['id_user', 'email', 'timestamp_reg']
    ml_ready_df = final_df.drop(columns=cols_to_drop)

    output_name = 'train_ML_READY.csv'
    ml_ready_df.to_csv(output_name, index=False)
    print(f"\n✅ ГОТОВО! Новий покращений датасет збережено як {output_name}")

if __name__ == "__main__":
    build_ml_ready_dataset()