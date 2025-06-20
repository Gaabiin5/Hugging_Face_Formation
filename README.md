# Hugging_Face_Formation

Here the [link](https://huggingface.co/learn/agents-course/unit0/introduction) of the formation if you want to follow it.


In this repo, I will put every notebook I will modify in order to keep a track of what I've done.
The folders follows the structure of the formation.


## Running code
I run the models locally thanks to Ollama. I use the version 0.9.1.

Without it the code must not work on your computer. There is a [page](https://huggingface.co/learn/agents-course/unit0/onboarding) in the formation who explained how to install ollama .

After installing it, you must pull the differents models you want to use. In my case I used the following models :

- openchat:latest
- phi4-mini:latest
- deepseek-coder:6.7b-instruct
- mistral instruct

### Usefuls commands to remind 

- ollama pull nameofthemodel

 (Pull the model locally so you can use it)

- ollama list 

(To see which models are loaded on the GPU)

- ollama rm nameofthemodel 

(To delete a model from the GPU storage)

- ollama run nameofthemodel 

(To launch the chat in the command window)

- nvitop 

(To verify if Ollama is running on the CPU or GPU)





## Summary and feedback

For each section, I will try to write a summary of the content and give feedback on the issues I encountered and possible improvements.
