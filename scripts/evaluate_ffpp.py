import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter
import pandas as pd
from PIL import Image

import torchvision
from torchvision import transforms

from pymilvus import (
    connections,
    Collection
)

sys.path.append('/home/manipcomnum/Bureau/fft_pro')
from train_encoder import DualSpectralViT_KAN

MODEL_PATH = "./fixed_incremental_model1/model1.pth"
FFPP_DATA_ROOT = "/home/manipcomnum/Bureau/fft_pro/face++"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FAKE_TECHNIQUES = [
    "DeepFake", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"
]

TOP_K_VALUES = [5, 10, 15]

print(" Loading model and setting up...")
model = DualSpectralViT_KAN(embed_dim=768, num_heads=12, n_bands=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print(" Connecting to Milvus...")
connections.connect("default", uri="/home/manipcomnum/Bureau/fft_pro/milvus/milvus.db")
collection = Collection("deepfake_embeddings")
collection.load()
print(f" Connected to Milvus - {collection.num_entities} embeddings available")

def extract_embedding(image_path):
    """Extract embedding for a single image"""
    try:
        img = torchvision.datasets.folder.default_loader(image_path)
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            fused_features, _, _ = model(tensor, return_cross_attention_features=True)
        
        return fused_features.cpu().numpy()[0].tolist()
    except Exception as e:
        print(f" Error processing {image_path}: {e}")
        return None

def search_similar_images(query_embedding, top_k=5):
    """Search for similar images in Milvus"""
    search_params = {"metric_type": "COSINE", "params": {"ef": 200}}
    
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["file_path", "technique", "label", "file_name"]
    )
    
    similar_images = []
    for hit in results[0]:
        similar_images.append({
            "similarity": 1 - hit.distance,
            "label": hit.entity.get("label"),
            "technique": hit.entity.get("technique"),
            "file_path": hit.entity.get("file_path")
        })
    
    return similar_images

def majority_vote(retrieved_images):
    """Perform majority voting on retrieved images"""
    if not retrieved_images:
        return 0.5, "Unknown"
    
    labels = [img["label"] for img in retrieved_images]
    label_counts = Counter(labels)
    

    fake_count = label_counts.get(1, 0)
    total_count = len(labels)
    fake_percentage = fake_count / total_count
    

    predicted_label = 1 if fake_percentage > 0.5 else 0
    
    return fake_percentage, predicted_label

def test_technique_complete_ffpp(technique, top_k=5, sample_size=None):
    """Test RAG performance on BOTH Real and Fake images for a specific technique"""
    print(f"\n Testing {technique} technique with top_{top_k}...")
    
    real_dir = os.path.join(FFPP_DATA_ROOT, "Real")
    fake_dir = os.path.join(FFPP_DATA_ROOT, "Fake", technique)
    

    if not os.path.exists(fake_dir):

        alternative_names = {
            "DeepFake": "Deepfakes",
            "FaceShifter": "Faceshifter", 
            "FaceSwap": "Faceswap"
        }
        if technique in alternative_names:
            fake_dir = os.path.join(FFPP_DATA_ROOT, "Fake", alternative_names[technique])
    
    if not os.path.exists(real_dir):
        print(f" Real directory not found: {real_dir}")
        return None
    if not os.path.exists(fake_dir):
        print(f" Fake directory not found for {technique}: {fake_dir}")
        return None
    

    real_images = [os.path.join(real_dir, f) for f in sorted(os.listdir(real_dir)) 
                   if f.endswith(('.jpg', '.png', '.jpeg'))]
    fake_images = [os.path.join(fake_dir, f) for f in sorted(os.listdir(fake_dir)) 
                   if f.endswith(('.jpg', '.png', '.jpeg'))]
    

    if sample_size and sample_size < len(real_images):
        real_images = list(np.random.choice(real_images, sample_size, replace=False))
    if sample_size and sample_size < len(fake_images):
        fake_images = list(np.random.choice(fake_images, sample_size, replace=False))
    
    print(f"   Found {len(real_images)} real images, {len(fake_images)} {technique} fake images")
    
    results = {
        'technique': technique,
        'top_k': top_k,
        'real_correct': 0,
        'real_total': len(real_images),
        'fake_correct': 0,
        'fake_total': len(fake_images),
        'total_correct': 0,
        'total_tested': 0
    }
    

    if len(real_images) > 0:
        for img_path in tqdm(real_images, desc=f"   Real {technique}", leave=False):
            embedding = extract_embedding(img_path)
            if embedding:
                retrieved = search_similar_images(embedding, top_k=top_k)
                fake_percentage, predicted_label = majority_vote(retrieved)
                
                if predicted_label == 0:
                    results['real_correct'] += 1
                    results['total_correct'] += 1
                results['total_tested'] += 1
    

    if len(fake_images) > 0:
        for img_path in tqdm(fake_images, desc=f"   Fake {technique}", leave=False):
            embedding = extract_embedding(img_path)
            if embedding:
                retrieved = search_similar_images(embedding, top_k=top_k)
                fake_percentage, predicted_label = majority_vote(retrieved)
                
                if predicted_label == 1:
                    results['fake_correct'] += 1
                    results['total_correct'] += 1
                results['total_tested'] += 1
    

    results['real_accuracy'] = results['real_correct'] / results['real_total'] if results['real_total'] > 0 else 0
    results['fake_accuracy'] = results['fake_correct'] / results['fake_total'] if results['fake_total'] > 0 else 0
    results['overall_accuracy'] = results['total_correct'] / results['total_tested'] if results['total_tested'] > 0 else 0
    
    print(f"    {technique}: {results['overall_accuracy']:.1%} overall "
          f"({results['real_accuracy']:.1%} real, {results['fake_accuracy']:.1%} fake)")
    
    return results

def discover_technique_names():
    """Discover actual technique names in the dataset"""
    fake_root = os.path.join(FFPP_DATA_ROOT, "Fake")
    if not os.path.exists(fake_root):
        print(f" Fake root directory not found: {fake_root}")
        return []
    
    techniques = [d for d in os.listdir(fake_root) 
                  if os.path.isdir(os.path.join(fake_root, d))]
    
    print(f" Discovered techniques: {techniques}")
    return techniques

def run_ffpp_comprehensive_test(sample_size=1000):
    """Run comprehensive testing on FF++ dataset"""
    print(" STARTING COMPREHENSIVE RAG TESTING ON FF++ DATASET")
    print("=" * 70)
    print(f" Using sample size: {sample_size} images per category")
    

    actual_techniques = discover_technique_names()
    if not actual_techniques:
        print(" No techniques found in dataset!")
        return []
    
    all_results = []
    
    for top_k in TOP_K_VALUES:
        print(f"\n TESTING WITH top_{top_k}")
        print("-" * 50)
        

        for technique in actual_techniques:
            results = test_technique_complete_ffpp(technique, top_k=top_k, sample_size=sample_size)
            if results:
                all_results.append(results)
    
    return all_results

def analyze_ffpp_results(all_results):
    """Analyze and display FF++ test results"""
    

    df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 100)
    print(" COMPREHENSIVE RAG TESTING RESULTS - FF++ DATASET")
    print("=" * 100)
    

    print("\n DETAILED RESULTS BY TECHNIQUE:")
    print("-" * 100)
    
    display_df = df.copy()
    display_df['Real_Accuracy'] = display_df['real_accuracy'].apply(lambda x: f"{x:.1%}")
    display_df['Fake_Accuracy'] = display_df['fake_accuracy'].apply(lambda x: f"{x:.1%}")
    display_df['Overall_Accuracy'] = display_df['overall_accuracy'].apply(lambda x: f"{x:.1%}")
    
    display_columns = ['technique', 'top_k', 'Real_Accuracy', 'Fake_Accuracy', 'Overall_Accuracy', 
                      'real_correct', 'real_total', 'fake_correct', 'fake_total']
    print(display_df[display_columns].to_string(index=False))
    

    print("\n AVERAGE PERFORMANCE BY TECHNIQUE:")
    print("-" * 70)
    
    technique_summary = df.groupby('technique').agg({
        'real_accuracy': 'mean',
        'fake_accuracy': 'mean',
        'overall_accuracy': 'mean',
        'real_correct': 'sum',
        'real_total': 'sum',
        'fake_correct': 'sum',
        'fake_total': 'sum'
    }).reset_index()
    
    technique_summary['Real_Accuracy'] = technique_summary['real_accuracy'].apply(lambda x: f"{x:.1%}")
    technique_summary['Fake_Accuracy'] = technique_summary['fake_accuracy'].apply(lambda x: f"{x:.1%}")
    technique_summary['Overall_Accuracy'] = technique_summary['overall_accuracy'].apply(lambda x: f"{x:.1%}")
    
    summary_columns = ['technique', 'Real_Accuracy', 'Fake_Accuracy', 'Overall_Accuracy']
    print(technique_summary[summary_columns].to_string(index=False))
    

    print("\n PERFORMANCE BY TOP_K VALUE:")
    print("-" * 60)
    
    top_k_summary = df.groupby('top_k').agg({
        'real_accuracy': 'mean',
        'fake_accuracy': 'mean',
        'overall_accuracy': 'mean'
    }).reset_index()
    
    top_k_summary['Real_Accuracy'] = top_k_summary['real_accuracy'].apply(lambda x: f"{x:.1%}")
    top_k_summary['Fake_Accuracy'] = top_k_summary['fake_accuracy'].apply(lambda x: f"{x:.1%}")
    top_k_summary['Overall_Accuracy'] = top_k_summary['overall_accuracy'].apply(lambda x: f"{x:.1%}")
    
    print(top_k_summary[['top_k', 'Real_Accuracy', 'Fake_Accuracy', 'Overall_Accuracy']].to_string(index=False))
    

    print("\n BEST PERFORMING TECHNIQUES:")
    print("-" * 50)
    
    if len(technique_summary) > 0:
        best_overall = technique_summary.loc[technique_summary['overall_accuracy'].idxmax()]
        best_fake = technique_summary.loc[technique_summary['fake_accuracy'].idxmax()]
        best_real = technique_summary.loc[technique_summary['real_accuracy'].idxmax()]
        
        print(f" Best Overall: {best_overall['technique']} - {best_overall['overall_accuracy']:.1%}")
        print(f" Best Fake Detection: {best_fake['technique']} - {best_fake['fake_accuracy']:.1%}")
        print(f" Best Real Detection: {best_real['technique']} - {best_real['real_accuracy']:.1%}")
    

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    detailed_file = f"rag_ffpp_complete_results_{timestamp}.csv"
    df.to_csv(detailed_file, index=False)
    print(f"\n Detailed results saved to: {detailed_file}")
    
    summary_file = f"rag_ffpp_technique_summary_{timestamp}.csv"
    technique_summary.to_csv(summary_file, index=False)
    print(f" Technique summary saved to: {summary_file}")
    
    return df

if __name__ == "__main__":
    print(" RAG TESTING FRAMEWORK - FF++ DATASET EVALUATION")
    print(f"Top_K values: {TOP_K_VALUES}")
    

    SAMPLE_SIZE = 1000
    
    print(f"Sample size per category: {SAMPLE_SIZE}")
    

    all_results = run_ffpp_comprehensive_test(sample_size=SAMPLE_SIZE)
    
    if all_results:

        df = analyze_ffpp_results(all_results)
        
        print(f"\n FF++ TESTING COMPLETED!")
        total_tested = df['total_tested'].sum()
        total_correct = df['total_correct'].sum()
        overall_accuracy = total_correct / total_tested if total_tested > 0 else 0
        
        print(f"   Total images tested: {total_tested}")
        print(f"   Total correct predictions: {total_correct}")
        print(f"   Overall accuracy: {overall_accuracy:.1%}")
        print(f"   Results saved to CSV files")
    else:
        print(" No test results generated. Check if FF++ data exists.")
