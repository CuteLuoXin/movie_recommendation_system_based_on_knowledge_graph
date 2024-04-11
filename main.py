if __name__ == '__main__':
    from KGCN.tool import construct_undirected_kg, get_adj_list
    from KGCN.model import KGCN_model
    from KGCN.train import train
    from data import kg_loader, data_process
    import tensorflow as tf
    # 加载ml-1m知识库数据，负样本比例选择40%
    (n_user, n_item, n_entity, n_relation, train_data, test_data, kg,
     topk_data) = data_process.pack_kg(kg_loader.ml1m_kg1m, negative_sample_threshold=4)

    neighbor_size = 16
    adj_entity, adj_relation = get_adj_list(construct_undirected_kg(kg),
                                            n_entity, neighbor_size)
    # 选择KGCN模型
    model = KGCN_model(n_user, n_entity, n_relation, adj_entity, adj_relation,
                       neighbor_size, iter_size=1, dim=16, l2=1e-7, aggregator='sum')

    train(model, train_data, test_data, topk_data,
          optimizer=tf.keras.optimizers.Adam(0.01), epochs=10, batch=512)
