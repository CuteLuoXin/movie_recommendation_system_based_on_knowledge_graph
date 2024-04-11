# 创建知识图谱三元组并将结果存储到 kg.txt 文件中
with open('ratings.dat', 'r',encoding='latin1') as ratings_file, open('users.dat', 'r',encoding='latin1') as users_file, open('movies.dat', 'r',encoding='latin1') as movies_file, open('kg.txt', 'w',encoding='latin1') as kg_file:
    # 处理 ratings.dat 文件
    for line in ratings_file:
        user_id, movie_id, rating, _ = line.strip().split('::')
        kg_file.write(f'User_{user_id}::rated::Movie_{movie_id}::{rating}\n')

    # 处理 users.dat 文件
    user_id2entity_id = {}
    for line in users_file:
        user_id, gender, age, occupation, _ = line.strip().split('::')
        entity_id = f'User_{user_id}'
        user_id2entity_id[user_id] = entity_id
        kg_file.write(f'{entity_id}::has_gender::{gender}\n')
        kg_file.write(f'{entity_id}::has_age::{age}\n')
        kg_file.write(f'{entity_id}::has_occupation::{occupation}\n')

    # 处理 movies.dat 文件
    item_id2entity_id = {}
    for line in movies_file:
        movie_id, title, genres = line.strip().split('::')
        entity_id = f'Movie_{movie_id}'
        item_id2entity_id[movie_id] = entity_id
        kg_file.write(f'{entity_id}::has_title::{title}\n')
        kg_file.write(f'{entity_id}::has_genres::{genres}\n')

# 将 item_id2entity_id 映射关系写入 item_id2entity_id.txt 文件中
with open('item_id2entity_id.txt', 'w',encoding='latin1') as item_mapping_file:
    for item_id, entity_id in item_id2entity_id.items():
        item_mapping_file.write(f'{item_id} {entity_id}\n')
