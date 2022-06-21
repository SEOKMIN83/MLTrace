'''
You should install alive-progress library in your terminal 
`$ pip3 install progress alive-progress tqdm`
'''
from alive_progress import alive_bar
import pandas as pd
import json
import shutil
import sidetable as stb


### Define create chunk list
def read_rawdata_return_chunklist(filename="raw data.txt", step=1000000) -> list:
    ## find correct start line
    f = open(filename, 'r')

    startline = 0  ## read raw data from this location

    while True:
        line = f.readline()
        startline += 1
        if (line.startswith('read') or line.startswith('write')):
            break
    f.close()
    print('startline is', startline)


    ## read raw data and create chunk list
    chunk = pd.read_csv(filename, names=['operation', 'address', 'size'], usecols=['operation', 'address'],
                        delim_whitespace=True, lineterminator='\n', skiprows=startline - 1, chunksize=step,
                        header=None, on_bad_lines='skip')
    chunk = list(chunk)  ## make a chunk list

    return chunk


## Define make 'page' column at each chunk
def make_pageNumber(df) -> pd.DataFrame:
    invalid_idx = df[df['operation'].str.contains('==')].index  ## extract invalid index
    df.drop(invalid_idx, inplace=True)  ## drop the invalid rows

    df['page'] = [int(i, 16) >> 12 for i in df['address']]  ## calculate page number (4KB)

    return df


## Define make 'logical time' column at each chunk
def make_logicalTime(df) -> pd.DataFrame:
    invalid_idx = df[df['operation'].str.contains('==')].index  ## extract invalid index
    df.drop(invalid_idx, inplace=True)  ## drop the invalid rows

    df.reset_index(inplace=True)  ## modify the DataFrame inplace (do not create a new object)
    df.rename(columns={'index': 'logical time'}, inplace=True)  ## modify the column name inplace

    return df


## Define make initial 'access count' columns
def make_initialaccessCount_eachChunk(df) -> pd.DataFrame():
    df_temp1 = df.groupby(['page'])['page'].count().reset_index(name='access count')
    # print(df_temp1.head(8))
    # print(df_temp1.shape[0])

    df_temp2 = df.loc[df.operation != 'write']  ## read
    df_temp2 = df_temp2.groupby(['page'])['page'].count().reset_index(name='access count')
    # print(df_temp2.head(8))
    # print(df_temp2.shape[0])

    df_temp3 = df.loc[df.operation == 'write']  ## write
    df_temp3 = df_temp3.groupby(['page'])['page'].count().reset_index(name='access count')
    # print(df_temp3.head(8))
    # print(df_temp3.shape[0])

    ## Descending sorting by 'access count'
    df_temp1.sort_values('access count', ascending=False, inplace=True)
    df_temp2.sort_values('access count', ascending=False, inplace=True)
    df_temp3.sort_values('access count', ascending=False, inplace=True)

    return df_temp1, df_temp2, df_temp3


## Define link all chunks into one chunk
def linking_chunks(input_chunk) -> pd.DataFrame:
    if len(input_chunk) == 1:  ## The number of chunk is 1
        num_lines = len(input_chunk)
        print("linking_chunks num_lines is " + str(num_lines))
        with alive_bar(num_lines, force_tty=True) as bar:
            df = input_chunk[0]
            bar()
    else:
        num_lines = len(input_chunk)
        print("linking_chunks num_lines is " + str(num_lines))
        df = pd.DataFrame()
        with alive_bar(num_lines, force_tty=True) as bar:
            for i in range(len(input_chunk)):
                df = df.append(input_chunk[i], ignore_index=True)
                bar()
    return df


## Define make final 'accumul count' column
def make_finalaccessCount(df, length) -> pd.DataFrame:
    num_lines = df.shape[0]
    print("make_finalaccessCount num_lines is " + str(num_lines))

    if length == 1:  ## The number of chunk is 1
        df['accumul count'] = df['access count']
    else:
        df = df.groupby(['page'])['access count'].sum().reset_index(name='accumul count')
        df.dropna(axis=0, inplace=True)  ## remove the rows that have NaN value

    df_xis_frequencyRanking = df.sort_values('accumul count', ascending=False)
    df_xis_pageNumber = df.sort_values('page', ascending=True)

    return df_xis_frequencyRanking, df_xis_pageNumber


## Define Descending Sorting by 'cumulative_percent' value
def sorting_by_cumulativePercent(df_total, df_read, df_write) -> pd.DataFrame:
    ## total access count
    df_total = df_total.stb.freq(['page'], value='accumul count')
    df_total.drop(['accumul count', 'cumulative_accumul count'], axis=1, inplace=True)
    df_total.sort_values('cumulative_percent', ascending=True, inplace=True)

    ## read access count
    df_read = df_read.stb.freq(['page'], value='accumul count')
    df_read.drop(['accumul count', 'cumulative_accumul count'], axis=1, inplace=True)
    df_read.sort_values('cumulative_percent', ascending=True, inplace=True)

    ## write access count
    df_write = df_write.stb.freq(['page'], value='accumul count')
    df_write.drop(['accumul count', 'cumulative_accumul count'], axis=1, inplace=True)
    df_write.sort_values('cumulative_percent', ascending=True, inplace=True)

    return df_total, df_read, df_write


## Define Real-time LRU Ranking
def LRU_rank(df, page_rank, read_count, write_count):
    num_lines = df.shape[0]  ## numeber of lines of DataFrame df

    with alive_bar(num_lines, force_tty=True) as bar:
        for index, row in df.iterrows():
            try:
                acc_rank = page_rank.index(row['page'])  ## extract the index of the currently referenced page
                _ = page_rank.pop(acc_rank)  ## ignore the return value of pop()
                page_rank.insert(0, row['page'])  ## renew the page ranking to 0

                if row['operation'] != 'write':  ## read
                    try:
                        read_count[acc_rank] += 1
                    except IndexError:
                        for i in range(len(read_count), acc_rank + 1):
                            read_count.insert(i, 0)
                        read_count[acc_rank] += 1
                else:  ## write
                    try:
                        write_count[acc_rank] += 1
                    except IndexError:
                        for i in range(len(write_count), acc_rank + 1):
                            write_count.insert(i, 0)
                        write_count[acc_rank] += 1
            except ValueError:  ## the first referenced page
                page_rank.insert(0, row['page'])
            bar()
    return page_rank, read_count, write_count


## Define Real-time LFU Ranking
def LFU_rank(df, page_rank, page_count_rank, read_count, write_count):
    num_lines = df.shape[0]

    with alive_bar(num_lines, force_tty=True) as bar:
        for index, row in df.iterrows():
            try:  ## already referenced page
                ## load current information
                curr_rank = page_rank.index(row['page'])  ## current page rank
                curr_count = page_count_rank[curr_rank]  ## current access count

                ## calculate the new access count of the page
                new_count = curr_count + 1
                ## find how many pages that have the same number of new access count
                num_same = page_count_rank.count(new_count)

                ## calculate the new rank of page
                new_rank = curr_rank - num_same

                ## renew the rank of page
                page_rank.insert(new_rank, row['page'])
                ## renew the access count of page
                page_count_rank.insert(new_rank, new_count)

                ## remove the invalid rank and access count
                del page_rank[curr_rank + 1]
                del page_count_rank[curr_rank + 1]

                ## case: read
                if row['operation'] != 'write':
                    try:
                        read_count[curr_rank] += 1
                    except IndexError:
                        for i in range(len(read_count), curr_rank + 1):
                            read_count.insert(i, 0)
                        read_count[curr_rank] += 1

                ## case: write
                else:
                    try:
                        write_count[curr_rank] += 1
                    except IndexError:
                        for i in range(len(write_count), curr_rank + 1):
                            write_count.insert(i, 0)
                        write_count[curr_rank] += 1

            except ValueError:  ## not yet referenced page
                page_rank.append(row['page'])  ## add to page_rank
                page_count_rank.append(0)  ## add to page_count_rank
            bar()
    return page_rank, page_count_rank, read_count, write_count


# *****************************************************************************************#
# *****************************************************************************************#
# *****************************************************************************************#
# ***************************** SAVE and LOAD DATA ****************************************#
# *****************************************************************************************#
# *****************************************************************************************#
# *****************************************************************************************#

####################################################################################################
################################# SAVE DATA PARTS ##################################################
####################################################################################################

## Define save pre-processed data : with logical Time
def save_data_with_logicalTime(df, filename='output-logical.csv', index=0) -> None:
    if ("-logical-all.csv" in filename) and os.path.isfile(filename) and index == 0:
        pass
    else:
        if index == 0:  ##the first write case
            df.to_csv(filename, header=True, index=False, columns=['logical time', 'operation', 'page'],
                      mode='w')  ## encoding='utf-8-sig'
        else:
            df.to_csv(filename, header=True, index=False, columns=['logical time', 'operation', 'page'],
                      mode='a')  ## encoding='utf-8-sig'


## Define save pre-processed data : with final access count
def save_data_with_accessCount(df, filename='output-access.csv') -> None:
    if os.path.isfile(filename):
        pass
    else:
        df.to_csv(filename, header=True, index=False, columns=['page', 'accumul count'], mode='w')


## Define save pre-processed data : popularity checkpoints
def save_popularity_checkpoints(df_xis_frequencyRanking, df_xis_pageNumber, filename, distinction='') -> None:
    name = filename + distinction  ## e.g. desktop-gqview-or-photo + -randw
    if os.path.isfile(name + "-xis_frequencyRanking.json"):
        pass
    else:
        df_xis_frequencyRanking.to_json(name + "-xis_frequencyRanking.json")

    if os.path.isfile(name + "-xis_pageNumber.json"):
        pass
    else:
        df_xis_pageNumber.to_json(name + "-xis_pageNumber.json")


## Define save pre-rpocessed data : cumulative_percent checkpoints
def save_cumul_checkpoints(df, filename, distinction=' ') -> None:
    name = filename + distinction  ## e.g. desktop-gqview-or-photo + -randw
    if os.path.isfile(name + "-cumulative_percent.json"):
        pass
    else:
        df.to_json(name + "-cumulative_percent.json")


## Define save pre-processed data: lru checkpoints
def save_lru_checkpoints(page_rank, read_count, write_count, filename, num) -> None:
    if os.path.isfile(filename + '-lru-checkpoint' + str(num) + '.json'):
        pass
    else:
        save = {"page_rank": page_rank,
                "read_count": read_count,
                "write_count": write_count}
        with open(filename + '-lru-checkpoint' + str(num) + '.json', 'w', encoding='utf-8') as f:
            ## indent 2 is not needed but makes the file human-readable
            json.dump(save, f, indent=2)


## Define save pre-processed data: lfu checkpoints
def save_lfu_checkpoints(page_rank, page_count_rank, read_count, write_count, filename, num) -> None:
    if os.path.isfile(filename + '-lfu-checkpoint' + str(num) + '.json'):
        pass
    else:
        save = {"page_rank": page_rank,
                "page_count_rank": page_count_rank,
                "read_count": read_count,
                "write_count": write_count}
        with open(filename + '-lfu-checkpoint' + str(num) + '.json', 'w', encoding='utf-8') as f:
            ## indent 2 is not needed but makes the file human-readable
            json.dump(save, f, indent=2)


####################################################################################################
################################# LOAD DATA PARTS ##################################################
####################################################################################################

## Define load of popularity checkpoints
def load_popularity_checkpoints(filename, distinction='') -> pd.DataFrame:
    name = filename + distinction
    df_xis_frequencyRanking = pd.read_json(name + '-xis_frequencyRanking.json')
    df_xis_pageNumber = pd.read_json(name + '-xis_pageNumber.json')
    return df_xis_frequencyRanking, df_xis_pageNumber


## Define load of cumulative_percent checkpoints
def load_cumul_checkpoints(filename, distinction=' ') -> pd.DataFrame:
    name = filename + distinction
    df = pd.read_json(name + "-cumulative_percent.json")
    return df


## Define load of lru checkpoints
def load_lru_checkpoints(filename, num):
    with open(filename + '-lru-checkpoint' + str(num) + '.json', 'r') as f:
        load = json.load(f)
        page_rank = load['page_rank']
        read_count = load['read_count']
        write_count = load['write_count']

    return page_rank, read_count, write_count


## Define load of lfu checkpoints
def load_lfu_checkpoints(filename, num):
    with open(filename + '-lfu-checkpoint' + str(num) + '.json', 'r') as f:
        load = json.load(f)
        page_rank = load['page_rank']
        page_count_rank = load['page_count_rank']
        read_count = load['read_count']
        write_count = load['write_count']

    return page_rank, page_count_rank, read_count, write_count


##################################################################################################
########################################## THIS IS MAIN PART #####################################
##################################################################################################
"""
Usage:  python3 create_and_save_preprocessed_data.py gqview.txt desktop-gqview-or-photo
"""
if __name__ == "__main__":
    import sys

    input_filename = sys.argv[1]  ## get filename from command line e.g. gqview.txt
    chunk = read_rawdata_return_chunklist(input_filename)  ## arg->step: default is 1 million
    num_of_chunklist = len(chunk)  ## how many chunks in chunklist
    output_filename = sys.argv[2]  ## get filename from command line e.g. desktop-gqview-or-photo

    import os

    ## get current diectory path
    current_directory = os.getcwd() ## /home/sm_ple38/PycharmProjects/AIWorkloads

    ## create path of the checkpoints
    output_save_folder_path = '/home/sm_ple38/PycharmProjects/AIworkloads/' + '/checkpoints/'
    output_path = os.path.join(output_save_folder_path, output_filename)

    ## check the path of checkpoints and create the path if not exists
    if not os.path.exists(output_save_folder_path):
        os.mkdir(output_save_folder_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ## create 'page' column at each chunk
    for i in range(len(chunk)):
        chunk[i] = make_pageNumber(chunk[i])

    #################################
    ## Apply real-time LRU ranking###
    #################################
    lru_page_rank = []
    lru_read_count = []
    lru_write_count = []

    os.chdir(output_path)  ## cd output_path
    for i in range(len(chunk)):
        ### if - else 부분 논리적 수정 필요할 것 같음
        if os.path.isfile(output_filename + '-lru-checkpoint' + str(i) + '.json'):
            pass
        else:
            if i > 0:
                lru_page_rank, lru_read_count, lru_write_count = load_lru_checkpoints(output_filename, i - 1)
            lru_page_rank, lru_read_count, lru_write_count = LRU_rank(chunk[i], lru_page_rank, lru_read_count,
                                                                  lru_write_count)
            save_lru_checkpoints(lru_page_rank, lru_read_count, lru_write_count, output_filename, i)
            print("save %d lru checkpoint" % i)
    os.chdir(current_directory)  ## cd curr_directory

    #################################
    ## Apply real-time LFU ranking###
    #################################
    lfu_page_rank = []
    lfu_page_count_rank = []
    lfu_read_count = []
    lfu_write_count = []

    os.chdir(output_path)  ## cd output_path
    for i in range(len(chunk)):
        ### if - else 부분 논리적 수정 필요할 것 같음
        if os.path.isfile(output_filename + '-lfu-checkpoint' + str(i) + '.json'):
            pass
        else:
            if i > 0:
                lfu_page_rank, lfu_page_count_rank, lfu_read_count, lfu_write_count = load_lfu_checkpoints(output_filename,
                                                                                                       i - 1)
            lfu_page_rank, lfu_page_count_rank, lfu_read_count, lfu_write_count = LFU_rank(chunk[i], lfu_page_rank,
                                                                                       lfu_page_count_rank,
                                                                                       lfu_read_count, lfu_write_count)
            save_lfu_checkpoints(lfu_page_rank, lfu_page_count_rank, lfu_read_count, lfu_write_count, output_filename, i)
            print("save %d lfu checkpoint" % i)
    os.chdir(current_directory)  ## cd curr_directory

    ## logcial time - page number
    gnuplot_input_folder_path = current_directory + '/gnuplot-input/'
    gnuplot_input_file_path = os.path.join(gnuplot_input_folder_path, output_filename)

    if not os.path.exists(gnuplot_input_folder_path):
        os.mkdir(gnuplot_input_folder_path)
    if not os.path.exists(gnuplot_input_file_path):
        os.mkdir(gnuplot_input_file_path)

    os.chdir(gnuplot_input_file_path)  ## $cd gnuplot-input/output_filename/
    for i in range(len(chunk)):
        chunk[i] = make_logicalTime(chunk[i])  ## create 'logical time' column at each chunk
        save_data_with_logicalTime(chunk[i], output_filename + '-logical-all.csv', i)  ## make whole data to one file
        save_data_with_logicalTime(chunk[i], output_filename + '-' + str(i) + '.csv', 0)  ## make whole data to several files
    os.chdir(current_directory)

    source_path = os.path.join(gnuplot_input_folder_path, output_filename)  ## /gnuplot-input/output_filename
    dest_path = os.path.join(gnuplot_input_folder_path, 'all-csv')  ## /gnuplot-input/all-csv
    '''
        gnuplot_input_fil_path + '/' + output_filname is /home/sm_ple38/PycharmProjects/AIworkloads/gnuplot-input//-logical-all.csv
        => WRONG!!!! PATH 
    '''
    # shutil.copy(gnuplot_input_file_path + '/' + output_filename + '-logical-all.csv', 'all-csv')
    shutil.copy(source_path + '/' + output_filename + '-logical-all.csv', dest_path)

    ## popularity : access count
    total_chunk = []
    read_chunk = []
    write_chunk = []

    ## calculate access count per page for each type of operation(randw, r, w) in  each chunk
    for i in range(len(chunk)):
        tmp1, tmp2, tmp3 = make_initialaccessCount_eachChunk(chunk[i])

        if tmp1.empty == False:
            total_chunk.append(tmp1)
        if tmp2.empty == False:
            read_chunk.append(tmp2)
        if tmp3.empty == False:
            write_chunk.append(tmp3)

    ## Linking each chunk
    linked_total_chunk = linking_chunks(total_chunk)
    linked_read_chunk = linking_chunks(read_chunk)
    linked_write_chunk = linking_chunks(write_chunk)

    popul_total, pagenum_total = make_finalaccessCount(linked_total_chunk, len(total_chunk))
    popul_read, pagenum_read = make_finalaccessCount(linked_read_chunk, len(read_chunk))
    popul_write, pagenum_write = make_finalaccessCount(linked_write_chunk, len(write_chunk))

    ## cumulative_percent
    cumulPercent_total, cumulPercent_read, cumulPercent_write = sorting_by_cumulativePercent(popul_total, popul_read,
                                                                                             popul_write)

    ## save checkpoints
    os.chdir(output_path)  ## cd output_path
    save_popularity_checkpoints(popul_total, pagenum_total, output_filename, '-randw')
    save_popularity_checkpoints(popul_read, pagenum_read, output_filename, '-r')
    save_popularity_checkpoints(popul_write, pagenum_write, output_filename, '-w')

    save_cumul_checkpoints(cumulPercent_total, output_filename, '-randw')
    save_cumul_checkpoints(cumulPercent_read, output_filename, '-r')
    save_cumul_checkpoints(cumulPercent_write, output_filename, '-w')

    os.chdir(current_directory)  ## restore current directory

    print('Making Pre-Processed data files is DONE!')
    print('Check your directory!\n\n')
