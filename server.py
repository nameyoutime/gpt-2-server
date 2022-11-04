from urllib import request
from flask import Flask, jsonify, request
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from flask_cors import CORS
path = './gpt2-vietnamese/'
tokenizer = GPT2Tokenizer.from_pretrained(path, local_files_only=True)
model = GPT2LMHeadModel.from_pretrained(path, local_files_only=True)
app = Flask(__name__)
CORS(app)


@app.route('/generate', methods=['POST'])
def gen_text():
    json = request.get_json()
    print(json)

    text = json['text']
    
    max_length = json.get('max_length' , None) or 100
    min_length = json.get('min_length' , None) or 10
    do_sample = json.get('do_sample' , None) or True
    top_k = json.get('top_k' , None) or 50
    num_beams = json.get('num_beams' , None) or 5
    early_stopping = json.get('early_stopping' , None) or True
    no_repeat_ngram_size = json.get('no_repeat_ngram_size' , None) or 2
    
    num_return_sequences = json.get('num_return_sequences' , None) or 1
    print(max_length)


    # print(text)
    input_ids = tokenizer.encode(text, return_tensors='pt')
    sample_outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id,
                                    do_sample=do_sample,
                                    max_length=max_length,
                                    min_length=min_length,
                                    top_k=top_k,
                                    num_beams=num_beams,
                                    early_stopping=early_stopping,
                                    no_repeat_ngram_size=no_repeat_ngram_size,
                                    num_return_sequences=num_return_sequences)
    array = []
    for i, sample_output in enumerate(sample_outputs):
        print(">> Generated text {}\n\n{}".format(
            i+1, tokenizer.decode(sample_output.tolist())))
        array.append(tokenizer.decode(sample_output.tolist()))
    print('\n---')
    print(array[0])

    return jsonify({'text': array})


@app.route('/test', methods=['GET'])
def get_test():
    text = "Việt Nam là quốc gia có"
    return jsonify({'tasks': text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
