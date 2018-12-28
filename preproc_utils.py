def ingest_data(directory, file_pattern, columns_to_keep, filter_rule,
                        out=None, row_limit=-1, columns_to_keep_and_ignore=[]):
    """read in files whose names match a regex pattern, filter out 
    unwanted rows and columns and construct a big dataset.
    Then shuffle the data and limit the rows
    """
    first = True
    regex = re.compile(file_pattern)
    
    for root, dirs, files, in os.walk(directory):
        print(root)
        print(dirs)
        print(files)
        for filename in files:
            if not regex.match(filename) == None:
                path = os.path.join(root, filename)
                pa("adding this file:{0}\n".format(filename), out)
                one_df = pd.read_csv(path, error_bad_lines=False, encoding = "ISO-8859-1", dtype=str)

                # trim whitespace from column names
                original_col = one_df.columns
                new_col = []
                for col in original_col:
                    new_col.append(col.strip())
                one_df.columns = new_col

                filtered_df = one_df

                if first:
                    data = filtered_df
                else:
                    data = data.append(filtered_df)

                del one_df
                del filtered_df
                first = False
            
    if first == True:
        raise IOError("no files in the search_directory matched this pattern")      
    if not row_limit == -1:
        data = randomize_and_sample_dataframe(data, row_limit)
        pa("shape after shuffling and limiting:{0}".format(data.shape), out)  
    pa('final shape:{0}'.format(data.shape), out)   
    return data

def randomize_and_sample(df, row_limit):
    """this should be applied to your
    training and validation rows
    """
    row_cnt = df.shape[0]
    shuffle = np.random.permutation(np.arange(row_cnt))
    df = df.iloc[shuffle]
    df = df[:row_limit]
    return df


def print_append(message, out=None):
    """stands for 'print append'
    this function was created as a workaround for the problem
    of both printing to stdout in the ipython notebook AND
    writing out to a README. There might be a better way to
    do this: this was the fastest way that was not too messy
    """
    print(message)
    if out:
        out.append(message)

def push_slack_training_update(model_name, message_list, celebrate_ind):
    """push a message to Slack with user model_name
    and optional celebration
    """
    index = sum([ord(c)**2 for c in model_name]) % len(slack_emojis)
    emoji = slack_emojis[index]
    if celebrate_ind:
        text = ':tada: ' + '\n:tada: '.join(message_list)
    else:
        text = '\n'.join(message_list)
    json_dict={'text': text, 'icon_emoji': emoji, 'username': model_name}
    r = requests.post("https://hooks.slack.com/services/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",
                      json=json_dict)


def split_rows_into_sets(df, id_col_name):
    """split the dataframe into your training, validation
    and testing sets by an id column (usually session)
    """
    train_set = ["0","1","2","3","4","5","6","7"]
    vldt_set = ["8"]
    test_set = ["9"]
    
    # examine the fifth digit of the id to split the data
    def get_fifth_digit(int_set):
        return df.loc[list(map(lambda x: str(x)[4:5] in int_set, df[id_col_name]))]  
    train_df = get_fifth_digit(train_set)
    vldt_df = get_fifth_digit(vldt_set)
    test_df = get_fifth_digit(test_set)
    return train_df, vldt_df, test_df


def vectorize_char_data(char_data, d, maxlen):
    """['ham','gunk'] becomes 
    [[0 0 ... 47 3 5 6 47 ] [ 0 0 ... 47 7 8 9 10 47 ]]
    """
    char_vectors = []
    for i, example in enumerate(char_data):
        example = "\n" + str(example) + "\n" 
        # convert search argument to list of characters
        example = list(example)
        seq = []
        for j, character in enumerate(example):
            if character in d:
                seq.append(d[character])
            else:
                seq.append(d['UNKNOWN'])
        char_vectors.append(seq)
        #if i % 100000 == 0:
        #    print(i)
    char_vectors = np.array(char_vectors)
    char_vectors = sequence.pad_sequences(char_vectors, maxlen=maxlen)
    return char_vectors

def vectorize_targ_data(targ_data, d):
    """['123','456','789'] becomes 
    [[False,True,...False] [True,False,...False] [False,False,...True]]
    """
    targ_vectors = np.zeros((len(obj_data), len(d)), dtype=np.bool)
    for i, example in enumerate(obj_data):
        if example in d:
            j = d[example]
        else:
            # j = 1
            # ^ this is a hack because OS 3mil was not trained with 
            # a label of UNKNOWN included, only use the above if you need to
            # all future models should include an 'UNKNOWN' value in their
            # target label dictionary/ies
            j = d["UNKNOWN"]
        targ_vectors[i, j] = 1
        if i % 100000 == 0:
            print(i)
    return targ_vectors
