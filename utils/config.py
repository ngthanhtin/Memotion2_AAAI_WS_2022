import torch


class CFG:
    apex           = False
    max_len        = 25
    model_name     = 'multihop' # san, cnnbert_concat, cnnbert_san
    cnn_type       = 'efficientnetv2-s'
    
    scheduler      = 'CosineAnnealingWarmRestarts' 
    epochs         = 15
    batch_size     = 5 # 4 is for emotion, 5 is for sentiment
    learning_rate  = 2e-4
    weight_decay   = 1e-5
    min_lr         = 1e-6

    ##MODEL PARAMS##
    max_len        = 35
    embedding_dim  = 200
    units          = 256 #dimention of hidden layer
    hidden_d       = 512
    hidden_d2      = 100
    n_sentiment_classes = 3 # sentiment (positive, negative, neutral)
    n_emotion_classes = 2 # 2 for each types(humour, sarcasm, offensive, motivational)
    n_intensity_classes = [4,4,4,2]
    n_layers       = 1
    dropout        = 0.2
    epochs         = 25

    T_0            = 50
    T_max          = 4  
    seed           = 42
    n_fold         = 5
    trn_fold       = [0]
    train          = True

    word_embedding = './pretrained_embedding/weights_matrix2.npy'

    train_path     = '../memotion2/cleaned_text_train_images/'
    test_path      = '../memotion2/cleaned_text_val_images/'
    train_csv_path = '../memotion2/memotion_train.csv'
    test_csv_path  = '../memotion2/memotion_val.csv'
    
    tokenizer_path = './tokenizers/tokenizer2.pth'
    sentiment_model= './multihop_fold0test_sentiment_best.pth'
    emotion_model  = './temp_best/multihop_fold0_emotion_best.pth'
    intensity_model= './multihop_fold0_intensity_best.pth'
    device         = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    class_weight_gradient = torch.tensor([4.875,1,3]).to(device)

    class_weight_gradient_humour = torch.tensor([2,1]).float().to(device)
    class_weight_gradient_sarcasm = torch.tensor([2,1]).float().to(device)
    class_weight_gradient_offensive = torch.tensor([1.25,2]).float().to(device)
    class_weight_gradient_motivation = torch.tensor([1,1.25]).float().to(device)

    class_weight_gradient_intensity_humour = torch.tensor([0.08,0.27,0.52,0.13]).float().to(device) # 3.99,1,1.96,6.65
    class_weight_gradient_intensity_sarcasm = torch.tensor([1,2.2,3.62,12.86]).float().to(device)
    class_weight_gradient_intensity_offensive = torch.tensor([1,4.68,9.79,28.47]).float().to(device)
    class_weight_gradient_intensity_motivation = torch.tensor([1,23.48]).float().to(device)