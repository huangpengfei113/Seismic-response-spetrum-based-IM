from functions import *
import math


def trans_csv(list1, title1, path1):
    new_list = [title1]
    new_list += [[sum([jj[ii] for jj in list1])/30] for ii in range(len(list1[0]))]
    write_csv(new_list, path1)


if __name__ == '__main__':
    path_dm = r'E:\demand measure.csv'  #File path of demand measure (layer drift) of subway station, ending with ".csv"
    path_spe_acc = r'E:\acceleration response spetrum.csv'  #File path of acceleration response spetrum, ending with ".csv"
    path_spe_vel = r'E:\velocity response spetrum.csv'  #File path of velocity response spetrum, ending with ".csv"
    path_spe_disp = r'E:\displacement response spetrum.csv'  #File path of displacement response spetrum, ending with ".csv"

    # Get DM
    data_dm = read_csv_data(path_dm)
    data_name = [i[0] for i in data_dm[1:]]
    data_col1 = [float(i[1]) for i in data_dm[1:]]
    data_col2 = [float(i[2]) for i in data_dm[1:]]
    # data_col_max = [math.log(max(ii/5.45, jj/6.07, kk/6.29)) for ii, jj, kk in zip(data_col1, data_col2, data_col3)]
    # 4.25, 5.92
    data_col_1 = [math.log(ii/5.45) for ii in data_col1]
    data_col_2 = [math.log(ii/6.07) for ii in data_col2]
    # Get seismic response spectrum
    data_spe_acc = read_csv_data(path_spe_acc)
    data_spe_acc = [[float(ii)/9.8/2 for ii in jj[2:]] for jj in data_spe_acc[1:]]

    data_spe_vel = read_csv_data(path_spe_vel)
    data_spe_vel = [[float(ii)/10 for ii in jj[2:]] for jj in data_spe_vel[1:]]

    data_spe_disp = read_csv_data(path_spe_disp)
    data_spe_disp = [[float(ii)/10 for ii in jj[2:]] for jj in data_spe_disp[1:]]

    max1 = max(data_col_1)
    max3 = max([max(i) for i in data_spe_acc])
    max4 = max([max(i) for i in data_spe_vel])
    max5 = max([max(i) for i in data_spe_disp])

    data_col1 = array_change(data_col_1)
    data_col2 = array_change(data_col_2)
    data_spe_disp = array_change2(data_spe_disp)
    data_spe_vel = array_change2(data_spe_vel)
    data_spe_acc = array_change2(data_spe_acc)

    training_data = []
    all_val_loss = {'col1-acc': []}
    all_train_loss = {'col1-acc': []}
    for try_i in range(30):
        col1, col2, col3, spec_acc, spec_vel, spec_disp, get_name = shuffle_numpy(data_col1, data_col2, data_col2,
                                                                                  data_spe_acc, data_spe_vel, data_spe_disp, data_name)
        len_input = len(data_spe_acc[0])
        training_data.append(get_name)

        pre_real = []
        for col_i, dm in zip([col1], ['col1']):
            for spec_i, im in zip([spec_acc], ['acc']):
                train_ge = gene_mlp(spec_i, col_i, 0, 1200, shuffle=True)
                vali_ge = gene_mlp(spec_i, col_i, 1200, None, shuffle=True)

                input_spec = Input(shape=(len_input,), name='spec_inp')
                spec_part1 = layers.Dense(1, activation=return_log, name='spec_mid', use_bias=False)(input_spec)
                spec_part4 = layers.Dense(1, activation='linear', name='spec_out')(spec_part1)

                network = models.Model(input_spec, spec_part4)
                network.compile(optimizer='adam', loss='mae')

                network.summary()

                history = network.fit_generator(train_ge, steps_per_epoch=150, epochs=40, validation_data=vali_ge,
                                                validation_steps=40, verbose=2)

                loss = history.history['loss']
                val_loss = history.history['val_loss']
                all_val_loss[dm+'-'+im].append(val_loss)
                all_train_loss[dm+'-'+im].append(loss)
                
                test_ge = gene_mlp_judge(spec_i, col_i, 1200, None, batch_size=len(spec_i)-1)
                for input, output, judge in test_ge:
                    pre_output = network.predict(input)
                    real_output = output
                    break
                pre_real.append([i for i in real_output])
                pre_real.append([i[0] for i in pre_output])
                
        pre_real_title = [['acc-col1实际', 'acc-col1预测']]
        path_temp1 = os.path.join(r'path to save the comparison data', 'Comparison in trial-{}.csv'.format(str(try_i)))
        trans_csv(pre_real, pre_real_title, path_temp1)

    path_vali_loss = r'E:\vali_loss_all.csv'  # path to save the loss value on test set
    write_csv(all_val_loss['col1-acc'], path_vali_loss)
    
    for key_i in all_val_loss.keys():
        path1 = os.path.join(r'E:\file to save average loss', key_i+'-valid.csv')
        trans_csv(all_val_loss[key_i], ['average loss'], path1)
    for key_i in all_train_loss.keys():
        path1 = os.path.join(r'E:\file to save average loss', key_i+'-train.csv')
        trans_csv(all_train_loss[key_i], ['average loss'], path1)
    
