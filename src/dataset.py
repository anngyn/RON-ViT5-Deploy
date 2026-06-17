"""Dataset classes for ReceiptVQA."""
import random
import pandas as pd
import torch
from torch.utils.data import Dataset


class TextOnlyVQADataset(Dataset):
    """Clean dataset for evaluation."""

    def __init__(self, qa_df, ocr_df, tokenizer, batch_process=128,
                 max_input_length=256, max_output_length=64):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.feature = ["input_ids", "src_attention_mask", "label_ids",
                        "label_attention_mask", "image_id", "question_id"]
        self.data = {key: [] for key in self.feature}

        dataframe = pd.merge(qa_df, ocr_df[['image_id', 'texts']], on='image_id', how='inner')
        self.batch_processing(dataframe, batch_process)

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.data['input_ids'][idx], dtype=torch.long),
            'src_attention_mask': torch.tensor(self.data['src_attention_mask'][idx], dtype=torch.long),
            'label_ids': torch.tensor(self.data['label_ids'][idx], dtype=torch.long),
            'label_attention_mask': torch.tensor(self.data['label_attention_mask'][idx], dtype=torch.long),
            'image_id': torch.tensor(self.data['image_id'][idx], dtype=torch.long),
            'question_id': torch.tensor(self.data['question_id'][idx], dtype=torch.long),
        }

    def batch_processing(self, dataframe, batch):
        self.data['image_id'] = list(dataframe['image_id'])
        self.data['question_id'] = list(dataframe['question_id'])

        for i in range(0, len(dataframe), batch):
            input_ids, src_attention_mask, label_ids, label_attention_mask = self.create_features(
                dataframe['question'][i:i+batch],
                dataframe['texts'][i:i+batch],
                dataframe['answer'][i:i+batch]
            )

            self.data['input_ids'] += input_ids
            self.data['src_attention_mask'] += src_attention_mask
            self.data['label_ids'] += label_ids
            self.data['label_attention_mask'] += label_attention_mask

    def create_features(self, ques, words, ans):
        inputs = [f"question: {q.strip()} context: {' '.join(ocr)}" for q, ocr in zip(ques, words)]
        outputs = [self.tokenizer.pad_token + a + self.tokenizer.eos_token for a in ans]

        encoding = self.tokenizer(inputs, add_special_tokens=True, max_length=self.max_input_length,
                                  padding="max_length", truncation=True)
        answer_encoding = self.tokenizer(outputs, add_special_tokens=False, max_length=self.max_output_length,
                                         padding="max_length", truncation=True)

        return encoding['input_ids'], encoding['attention_mask'], answer_encoding['input_ids'], answer_encoding['attention_mask']


class NoisyVQADataset(Dataset):
    """Dataset with noisy augmentation (clean + noisy samples)."""

    def __init__(self, qa_df, ocr_df, tokenizer, noise_generator,
                 augmentation_ratio=1.0, noise_types=['mixed'], noise_level=2,
                 batch_process=128, max_input_length=256, max_output_length=64,
                 include_clean=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.noise_generator = noise_generator
        self.noise_types = noise_types
        self.noise_level = noise_level
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.feature = ["input_ids", "src_attention_mask", "label_ids",
                        "label_attention_mask", "image_id", "question_id"]
        self.data = {key: [] for key in self.feature}

        dataframe = pd.merge(qa_df, ocr_df[['image_id', 'texts']], on='image_id', how='inner')

        # Clean samples
        if include_clean:
            clean_df = dataframe.copy()
            self.batch_processing(clean_df, batch_process, apply_noise=False)

        # Noisy samples
        n_augment = int(len(dataframe) * augmentation_ratio)
        noisy_df = dataframe.sample(n=n_augment, replace=True, random_state=42).reset_index(drop=True)
        self.batch_processing(noisy_df, batch_process, apply_noise=True)

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.data['input_ids'][idx], dtype=torch.long),
            'src_attention_mask': torch.tensor(self.data['src_attention_mask'][idx], dtype=torch.long),
            'label_ids': torch.tensor(self.data['label_ids'][idx], dtype=torch.long),
            'label_attention_mask': torch.tensor(self.data['label_attention_mask'][idx], dtype=torch.long),
            'image_id': torch.tensor(self.data['image_id'][idx], dtype=torch.long),
            'question_id': torch.tensor(self.data['question_id'][idx], dtype=torch.long),
        }

    def batch_processing(self, dataframe, batch, apply_noise=False):
        self.data['image_id'] += list(dataframe['image_id'])
        self.data['question_id'] += list(dataframe['question_id'])

        for i in range(0, len(dataframe), batch):
            texts = dataframe['texts'][i:i+batch]

            if apply_noise:
                noisy_texts = []
                for text_list in texts:
                    clean_text = ' '.join(text_list)
                    noise_type = random.choice(self.noise_types)
                    noisy_text = self.noise_generator.apply_noise(clean_text, noise_type, self.noise_level)
                    noisy_texts.append(noisy_text.split())
                texts = noisy_texts

            input_ids, src_attention_mask, label_ids, label_attention_mask = self.create_features(
                dataframe['question'][i:i+batch], texts, dataframe['answer'][i:i+batch]
            )

            self.data['input_ids'] += input_ids
            self.data['src_attention_mask'] += src_attention_mask
            self.data['label_ids'] += label_ids
            self.data['label_attention_mask'] += label_attention_mask

    def create_features(self, ques, words, ans):
        inputs = [f"question: {q.strip()} context: {' '.join(ocr)}" for q, ocr in zip(ques, words)]
        outputs = [self.tokenizer.pad_token + a + self.tokenizer.eos_token for a in ans]

        encoding = self.tokenizer(inputs, add_special_tokens=True, max_length=self.max_input_length,
                                  padding="max_length", truncation=True)
        answer_encoding = self.tokenizer(outputs, add_special_tokens=False, max_length=self.max_output_length,
                                         padding="max_length", truncation=True)

        return encoding['input_ids'], encoding['attention_mask'], answer_encoding['input_ids'], answer_encoding['attention_mask']


class PairedVQADataset(Dataset):
    """Paired clean/noisy samples for consistency loss."""

    def __init__(self, qa_df, ocr_df, tokenizer, noise_generator,
                 noise_types=['mixed'], noise_level=2,
                 batch_process=128, max_input_length=256, max_output_length=64):
        super().__init__()
        self.tokenizer = tokenizer
        self.noise_generator = noise_generator
        self.noise_types = noise_types
        self.noise_level = noise_level
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.clean_data = {"input_ids": [], "src_attention_mask": [], "label_ids": [], "label_attention_mask": []}
        self.noisy_data = {"input_ids": [], "src_attention_mask": [], "label_ids": [], "label_attention_mask": []}
        self.metadata = {"image_id": [], "question_id": []}

        dataframe = pd.merge(qa_df, ocr_df[['image_id', 'texts']], on='image_id', how='inner')

        # Process clean
        self.batch_processing(dataframe, batch_process, apply_noise=False, target_dict=self.clean_data)

        # Process noisy (same order)
        self.batch_processing(dataframe, batch_process, apply_noise=True, target_dict=self.noisy_data)

        # Metadata
        self.metadata['image_id'] = list(dataframe['image_id'])
        self.metadata['question_id'] = list(dataframe['question_id'])

    def __len__(self):
        return len(self.clean_data['input_ids'])

    def __getitem__(self, idx):
        return {
            'clean_input_ids': torch.tensor(self.clean_data['input_ids'][idx], dtype=torch.long),
            'clean_attention_mask': torch.tensor(self.clean_data['src_attention_mask'][idx], dtype=torch.long),
            'clean_label_ids': torch.tensor(self.clean_data['label_ids'][idx], dtype=torch.long),
            'noisy_input_ids': torch.tensor(self.noisy_data['input_ids'][idx], dtype=torch.long),
            'noisy_attention_mask': torch.tensor(self.noisy_data['src_attention_mask'][idx], dtype=torch.long),
            'noisy_label_ids': torch.tensor(self.noisy_data['label_ids'][idx], dtype=torch.long),
        }

    def batch_processing(self, dataframe, batch, apply_noise, target_dict):
        for i in range(0, len(dataframe), batch):
            texts = dataframe['texts'][i:i+batch]

            if apply_noise:
                noisy_texts = []
                for text_list in texts:
                    clean_text = ' '.join(text_list)
                    noise_type = random.choice(self.noise_types)
                    noisy_text = self.noise_generator.apply_noise(clean_text, noise_type, self.noise_level)
                    noisy_texts.append(noisy_text.split())
                texts = noisy_texts

            input_ids, src_attention_mask, label_ids, label_attention_mask = self.create_features(
                dataframe['question'][i:i+batch], texts, dataframe['answer'][i:i+batch]
            )

            target_dict['input_ids'] += input_ids
            target_dict['src_attention_mask'] += src_attention_mask
            target_dict['label_ids'] += label_ids
            target_dict['label_attention_mask'] += label_attention_mask

    def create_features(self, ques, words, ans):
        inputs = [f"question: {q.strip()} context: {' '.join(ocr)}" for q, ocr in zip(ques, words)]
        outputs = [self.tokenizer.pad_token + a + self.tokenizer.eos_token for a in ans]

        encoding = self.tokenizer(inputs, add_special_tokens=True, max_length=self.max_input_length,
                                  padding="max_length", truncation=True)
        answer_encoding = self.tokenizer(outputs, add_special_tokens=False, max_length=self.max_output_length,
                                         padding="max_length", truncation=True)

        return encoding['input_ids'], encoding['attention_mask'], answer_encoding['input_ids'], answer_encoding['attention_mask']
