import torch
import random
from typing import List, Dict, Tuple

class IOIDataset:
    """
    A robust and clean Indirect Object Identification (IOI) dataset generator
    tailored for gpt2-small and mechanistic interpretability.
    
    Generates balanced matching pairs of clean and corrupted prompts of identical length
    and returns precise token index tracking tensors.
    """
    
    # Strictly validated single-token names under GPT-2 tokenizer
    NAMES = [
        "John", "Mary", "David", "Alice", "Bob", "Kate", "Paul", "Jack", 
        "Jane", "Anne", "Bill", "Dave", "Sarah", "Mike", "Tom", "Harry", 
        "Lisa", "Amy"
    ]
    
    # Strictly validated single-token places and objects
    PLACES = ["store", "park", "school", "garden", "market", "office", "library", "kitchen", "house", "town"]
    OBJECTS = ["toy", "book", "bottle", "ring", "drink", "key", "coin", "cup", "bag", "pen"]
    
    # Standard templates. We support two styles to keep lengths perfectly uniform.
    TEMPLATES = {
        "A": "When {S} and {IO} went to the {place}, {S} gave a {object} to",
        "B": "Then, {S} and {IO} went to the {place}. {S} gave a {object} to"
    }
    
    def __init__(
        self, 
        model, 
        num_samples: int = 100, 
        template_style: str = "A", 
        seed: int = 42
    ):
        """
        Initializes the dataset generator.
        
        Args:
            model: The HookedTransformer model instance (used for tokenization).
            num_samples: Number of prompt pairs to generate.
            template_style: "A" or "B" (determines the standard template format).
            seed: Random seed for reproducibility.
        """
        self.model = model
        self.tokenizer = model.tokenizer
        self.num_samples = num_samples
        self.template_style = template_style
        
        # Dataset defense validation
        for name in self.NAMES:
            assert len(self.tokenizer.encode(" " + name)) == 1, f"Name '{name}' is multi-token!"
        
        # Set seed for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
        
        if template_style == "mixed":
            self.template = None
        else:
            self.template = self.TEMPLATES[template_style]
        
        # Generate raw string data
        self.samples = self._generate_samples()
        
        # Process and tokenize samples
        self.clean_prompts = [s["clean_prompt"] for s in self.samples]
        self.corrupted_prompts = [s["corrupted_prompt"] for s in self.samples]
        
        # Tokenize with forced prepend_bos=True
        # Shape: [batch, seq_len]
        self.clean_tokens = self.model.to_tokens(self.clean_prompts, prepend_bos=True)
        # Shape: [batch, seq_len]
        self.corrupted_tokens = self.model.to_tokens(self.corrupted_prompts, prepend_bos=True)
        
        self.batch_size, self.seq_len = self.clean_tokens.shape
        
        # Extract target token IDs
        # Shape: [batch]
        self.io_token_ids = torch.tensor(
            [self.tokenizer.encode(" " + s["io"])[0] for s in self.samples], 
            dtype=torch.long, 
            device=self.clean_tokens.device
        )
        # Shape: [batch]
        self.s_token_ids = torch.tensor(
            [self.tokenizer.encode(" " + s["s"])[0] for s in self.samples], 
            dtype=torch.long, 
            device=self.clean_tokens.device
        )
        # Shape: [batch]
        self.corrupted_token_ids = torch.tensor(
            [self.tokenizer.encode(" " + s["s_prime"])[0] for s in self.samples], 
            dtype=torch.long, 
            device=self.clean_tokens.device
        )
        
        # Track precise index positions
        self.S1 = torch.zeros(self.batch_size, dtype=torch.long, device=self.clean_tokens.device)
        self.IO = torch.zeros(self.batch_size, dtype=torch.long, device=self.clean_tokens.device)
        self.S2 = torch.zeros(self.batch_size, dtype=torch.long, device=self.clean_tokens.device)
        self.end = torch.zeros(self.batch_size, dtype=torch.long, device=self.clean_tokens.device)
        
        self._find_positions()

    def _generate_samples(self) -> List[Dict[str, str]]:
        """
        Generates distinct metadata tuples (S, IO, S', place, object) and string prompts.
        """
        samples = []
        for i in range(self.num_samples):
            # Select 3 unique names
            names = random.sample(self.NAMES, 3)
            s, io, s_prime = names[0], names[1], names[2]
            
            # Select a random place and object
            place = random.choice(self.PLACES)
            obj = random.choice(self.OBJECTS)
            
            if self.template_style == "mixed":
                # Alternate 50/50 between Template Style A and Template Style B
                template = self.TEMPLATES["A"] if i % 2 == 0 else self.TEMPLATES["B"]
            else:
                template = self.template
                
            # Clean prompt (S, IO, S)
            clean_prompt = template.format(S=s, IO=io, place=place, object=obj)
            
            # Corrupted prompt (S, S', S)
            # Replaces the IO name with a third distinct name S' to break the IOI circuit
            corrupted_prompt = template.format(S=s, IO=s_prime, place=place, object=obj)
            
            samples.append({
                "s": s,
                "io": io,
                "s_prime": s_prime,
                "place": place,
                "object": obj,
                "clean_prompt": clean_prompt,
                "corrupted_prompt": corrupted_prompt
            })
            
        return samples

    def _find_positions(self):
        """
        Locates the exact integer indices of S1, IO, S2, and end tokens in the tokenized prompts.
        Matches positions across clean and corrupted batches.
        """
        for i in range(self.batch_size):
            s_token_id = self.s_token_ids[i].item()
            io_token_id = self.io_token_ids[i].item()
            
            # Extract the tokens for sample i
            # Shape: [seq_len]
            tokens = self.clean_tokens[i]
            
            # Find occurrences of the subject name token
            s_indices = (tokens == s_token_id).nonzero(as_tuple=True)[0]
            
            # Find occurrences of the indirect object name token
            io_indices = (tokens == io_token_id).nonzero(as_tuple=True)[0]
            
            # Sanity check: We expect S to appear exactly twice, and IO to appear exactly once
            assert len(s_indices) == 2, (
                f"Sample {i}: Expected S name '{self.samples[i]['s']}' to appear twice, "
                f"found {len(s_indices)} times."
            )
            assert len(io_indices) == 1, (
                f"Sample {i}: Expected IO name '{self.samples[i]['io']}' to appear once, "
                f"found {len(io_indices)} times."
            )
            
            # Assign position indices
            self.S1[i] = s_indices[0].item()
            self.S2[i] = s_indices[1].item()
            self.IO[i] = io_indices[0].item()
            
            # The target prediction token index is the very last token in the prompt (typically the word "to")
            # Robust length-normalization by dynamically locating the last non-padding token:
            non_pad_indices = (tokens != self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            self.end[i] = non_pad_indices[-1].item()

    def __len__(self) -> int:
        return self.batch_size

    def __getitem__(self, idx: int) -> Dict:
        return {
            "clean_prompt": self.clean_prompts[idx],
            "corrupted_prompt": self.corrupted_prompts[idx],
            "clean_tokens": self.clean_tokens[idx],
            "corrupted_tokens": self.corrupted_tokens[idx],
            "S1": self.S1[idx],
            "IO": self.IO[idx],
            "S2": self.S2[idx],
            "end": self.end[idx],
            "io_token_id": self.io_token_ids[idx],
            "s_token_id": self.s_token_ids[idx]
        }
