# divide into train, val, and test
import os
import random
import natsort


if __name__ == "__main__":
    file_path = '/data/zhenghui.zhao/Dataset/Change Detection/WHU-CD-256/label'
    save_path = '/data/zhenghui.zhao/Dataset/Change Detection/WHU-CD-256'
    file_list = os.listdir(file_path)
    file_list = natsort.natsorted(file_list)



###  divide data ###
    '''
    # random.seed(0) for static division
    param file_list:            
    param train_rate:          
    param train_save_path:     
    param val_save_path:   
    '''
    train_rate = 0.7
    val_rate = 0.1

    train_save_path = '/data/zhenghui.zhao/Dataset/Change Detection/WHU-CD-256/list/train.txt'
    val_save_path = '/data/zhenghui.zhao/Dataset/Change Detection/WHU-CD-256/list/val.txt'
    test_save_path = '/data/zhenghui.zhao/Dataset/Change Detection/WHU-CD-256/list/test.txt'

    train_index = len(file_list) * train_rate
    val_index = len(file_list) * (train_rate + val_rate)


    random.seed(0)
    random.shuffle(file_list)


    if os.path.exists(train_save_path):
        print(f'{train_save_path} already exists! Please delete it first.')
    if os.path.exists(val_save_path):
        print(f'{val_save_path} already exists! Please delete it first.')
    if os.path.exists(test_save_path):
        print(f'{test_save_path} already exists! Please delete it first.')

    if not os.path.exists(train_save_path) and not os.path.exists(val_save_path) and not os.path.exists(test_save_path):
        print('Splitting datasets...')
        for i in range(len(file_list)):
            # train
            if i < train_index:
                with open(os.path.join(train_save_path), "a+", encoding="utf-8", errors="ignore") as f:
                    if i < train_index - 1:
                        f.write(file_list[i] + '\n')
                    else:
                        f.write(file_list[i])
            # val
            elif i >= train_index and i < val_index:
                with open(os.path.join(val_save_path), 'a+', encoding='utf-8', errors='ignore') as f:
                    if i < val_index - 1:
                        f.write(file_list[i] + '\n')
                    else:
                        f.write(file_list[i])
            # test
            else:
                with open(os.path.join(test_save_path), 'a+', encoding='utf-8', errors='ignore') as f:
                    if i < len(file_list) - 1:
                        f.write(file_list[i] + '\n')
                    else:
                        f.write(file_list[i])

        print(f'Train datasets was saved: {train_save_path}')
        print(f'Val datasets was saved: {val_save_path}')
        print(f'Test datasets was saved: {test_save_path}')
        print('Splitting datasets Finished!')