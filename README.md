<div align="center"> 
# ơShared task @ AAAI 2022: First Workshop on ​Multimodal Fact-Checking and Hate Speech Detection](https://aiisc.ai/defactify/memotion_2.html)
by [Thanh Tin Nguyen *](https://https://ngthanhtin.github.io/), Nhat Truong Pham, and Yong-Guk Kim. 

</div> 

# Create Tokenizer
python tokenizer.py, it will create a pth file, then put the generated file in `tokenizers/` folder.

*Note that, create tokenizer before doing the next steps.
# Create Pretrained Word Embedding
python utils/create_weights_matrix.py, it will create a npy file, then put the generated file in `pretrained_embedding/` folder.



# Trainning
`As for Sentiment Task (Task A):` </br>
python image_text/train_sentiment.py </br>
python unimodal_text/train_sentiment.py </br>
python unimodal_image/train_sentiment.py </br>
`As for Emotion Task (Task B):` </br>
python image_text/train_emotion.py </br>
python unimodal_text/train_emotion.py </br>
python unimodal_image/train_emotion.py </br>
`As for Intensity of Emotion Task (Task C):` </br>
python image_text/train_intensity.py </br>
python unimodal_text/train_intensity.py </br>
python unimodal_image/train_intensity.py </br>

# Inference
`As for Sentiment Task (Task A):` </br>
python image_text/inference_sentiment.py </br>
python unimodal_text/inference_sentiment.py </br>
python unimodal_image/inference_sentiment.py </br>
`As for Emotion Task (Task B):` </br>
python image_text/inference_emotion.py </br>
python unimodal_text/inference_emotion.py </br>
python unimodal_image/inference_emotion.py </br>
`As for Intensity of Emotion Task (Task C):` </br>
python image_text/inference_intensity.py </br>
python unimodal_text/inference_intensity.py </br>
python unimodal_image/inference_intensity.py </br>

# Result

# Submission
When running inference, it will create a file in `results/`. After inferencing 3 tasks, there will be 3 files in `results/`, then run `python utils/concat_submission.py` to generate `answer.txt`, zip it as `res.zip` to submit.

# Acknowledgement
[Multihop Attention for Meme Affect Analysis](https://github.com/LCS2-IIITD/MHA-MEME)

https://github.com/paritoshMahto07/Scene-Text-Detection-and-Recognition-
