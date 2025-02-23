import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

# Define paths here
MODEL_PATH = "/Users/nick/Developer/Personal_Project/Models/gpt2-trained"  # Path to your trained model directory
OUTPUT_PATH = "/Users/nick/Developer/Personal_Project/Models/coreml-trained/gpt2-trained.mlpackage"  # Output path for CoreML model

def convert_to_coreml(model_path, output_path):
    print(f"Loading model from {model_path}...")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create a wrapper class for the model
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids):
            # Use a simplified forward pass that avoids list comprehensions
            with torch.no_grad():
                outputs = self.model(input_ids)
                # Handle different output types explicitly
                if hasattr(outputs, 'logits'):
                    return outputs.logits
                elif isinstance(outputs, tuple):
                    return outputs[0]
                else:
                    return outputs
    
    # Wrap the model and trace it instead of scripting
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()
    
    # Create example input for tracing
    example_input = tokenizer("Example input", return_tensors="pt")["input_ids"]
    
    # Use tracing instead of scripting
    print("Tracing model...")
    try:
        traced_model = torch.jit.trace(wrapped_model, example_input)
    except Exception as e:
        print(f"Error during tracing: {e}")
        raise
    
    # Convert to CoreML format using tracing
    print("Converting to CoreML format...")
    try:
        # Define input shape with RangeDim for variable sequence length
        input_shape = ct.Shape(shape=(1, ct.RangeDim(1, 512)))  # Reduced max length for better compatibility
        
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="input_ids",
                    shape=input_shape,
                    dtype=np.int32
                )
            ],
            source="pytorch",
            minimum_deployment_target=ct.target.iOS16,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.ALL,
            convert_to="mlprogram"
        )
    except Exception as e:
        print(f"Error during CoreML conversion: {e}")
        raise
    
    # Add model metadata
    mlmodel.author = "Petal.AI"
    mlmodel.license = "MIT"
    mlmodel.short_description = "Fine-tuned GPT-2 model for conversational AI"
    mlmodel.version = "1.0"
    
    # Save the model
    print(f"Saving CoreML model to {output_path}...")
    mlmodel.save(output_path)
    print("Conversion complete!")

if __name__ == "__main__":
    convert_to_coreml(MODEL_PATH, OUTPUT_PATH)