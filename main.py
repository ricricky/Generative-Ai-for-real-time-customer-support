import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi.middleware.cors import CORSMiddleware
import sys

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging = logging.getLogger(__name__)


# Input model configuration
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium",force_download=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# Ensure proper tokenizer setup
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side='left' #( specific for  dialogpt for important tokens)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReplyRequest(BaseModel):
    message: str

class SmartReplyGenerator:
    def __init__(self, model, tokenizer):
        """
        Initialize the Smart Reply Generator
        
        Args:
            model: Pretrained language model
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_replies(self, message, max_length=25, num_replies=3):
        """
        Generate concise and straightforward replies
        
        Args:
            message (str): Input message
            max_length (int): Maximum length of generated replies
            num_replies (int): Number of replies to generate
        
        Returns:
            list: Generated reply candidates
        """
        if not message:
            return{"status":0,"message":"please input the text for replies"}
             
        
        # Ensure proper tokenization with chat model
        chat_history_ids = self.tokenizer.encode(message + self.tokenizer.eos_token, return_tensors="pt")
        
        # Generate replies with more verbose parameters
        with torch.no_grad():
            outputs = self.model.generate(
                chat_history_ids, 
                max_length=50,   
                num_return_sequences=num_replies, 
                no_repeat_ngram_size=2, 
                #top_p=0.9,      # less probability distribution 
                top_k=50,       # More token options
                temperature=0.2,  # less creativity more precise response 
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode the generated replies with more verbose decoding
        replies = []
        for output in outputs:
            reply = self.tokenizer.decode(output, skip_special_tokens=True)

           
            # Minimal cleaning
            cleaned_reply = reply.strip()
            
            # Additional filtering
            if cleaned_reply and len(cleaned_reply) > 3:
                # Remove the original message if it appears in the reply
                cleaned_reply = cleaned_reply.replace(message, '').strip()
                
                # Capitalize first letter and add period if needed
                if cleaned_reply:
                    cleaned_reply = cleaned_reply[0].upper() + cleaned_reply[1:]
                    if not cleaned_reply.endswith(('.', '!', '?')):
                        cleaned_reply += '.'
                
                replies.append(cleaned_reply)
        
        # Fallback if no replies generated
        if not replies:
            replies = ["I didn't quite catch that. Could you rephrase?"]
        
        return replies

# Initialize the smart reply generator
smart_reply_generator = SmartReplyGenerator(model, tokenizer)

@app.post("/generate_reply")
async def generate_reply(request: ReplyRequest):
    """
    Generate smart replies endpoint with extensive error handling
    """
    try:
        # Validate input
        if not request.message or len(request.message.strip()) == 0:
            return{"status":0,"message":"please input the text for replies"}
        
        # Generate replies
        logging.info(f"Generating replies for message: {request.message}")
        
        replies = smart_reply_generator.generate_replies(request.message)
        
        # Log generated replies
        logging.info(f"Generated replies: {replies}")
        
        return { 
            "status":1,
            "replies": replies
        }
    
    except ValueError as ve:
        # Handle specific value errors
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Log and handle unexpected errors
        logging.error(f"Error generating replies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
