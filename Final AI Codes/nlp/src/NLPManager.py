import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# For reformat_heading(text)
text_to_num = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "niner": "9",
    "nine": "9"
}

class NLPManager:
    def __init__(self):
        # Initialize the model here if necessary
        model_save_path = "fine-tuned-model30000"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_save_path)
        self.model.to(self.device)

        
    def convert_heading_to_number(self, heading_tokens):
        # Strip punctuation and convert text to numbers
        output = "".join([text_to_num.get(token.rstrip('.,'), token) for token in heading_tokens])

        while len(output) < 3:
            output = "0" + output

        return output

    def qa(self, context: str):
        # Define label mapping
        id2label = {0: 'O', 1: 'B-TOOL', 2: 'B-HEADING', 3: 'B-TARGET'}

        # Tokenize the input transcript
        tokens = context.replace('.', '').split()
        tokenized_inputs = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        word_ids = tokenized_inputs.word_ids()
        
        inputs = {k: v.to(self.device) for k, v in tokenized_inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs).logits
        
        # Convert the predictions to labels
        predictions = torch.argmax(outputs, dim=2).cpu().numpy()[0]

        # Align the labels with the original tokens
        aligned_labels = []
        previous_word_idx = None
        for word_idx, prediction in zip(word_ids, predictions):
            if word_idx is None or word_idx == previous_word_idx:
                continue
            aligned_labels.append((tokens[word_idx], id2label[prediction]))
            previous_word_idx = word_idx

        # Convert the aligned labels to the desired output format
        tool, heading, target = [], [], []
        for token, label in aligned_labels:
            if label == 'B-TOOL':
                tool.append(token)
            elif label == 'B-HEADING':
                heading.append(token)
            elif label == 'B-TARGET':
                target.append(token)
        
        
        
        output = {
            "heading": self.convert_heading_to_number(heading),
            "target": " ".join(target).strip(',.'),
            "tool": " ".join(tool).strip(',.'),
            
        }

        return output