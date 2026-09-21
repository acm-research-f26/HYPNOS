import torch
import os

#my new function
def save_prediction_summary(soft_preds, output_dir, prefix):
    #identifies the index of the highest probability acros all possible classes
    hard_preds = torch.argmax(soft_preds, dim=-1) 

#this portion determines the class names based on the prefix of the prediction head 
    if "sleepstages" in prefix:
        class_names = ["Deep", "Light", "REM", "Wake"]
    elif "respevents" in prefix: 
        class_names = ["No Event", "Event"]
    
    #identifies the predictions that are valid (no all NaN values) and uses them only for hard preds 
    valid_mask = ~torch.isnan(soft_preds).all(dim=-1)
    hard_preds = hard_preds[valid_mask] 

    total_patches = len(hard_preds)

    summary_path = os.path.join(output_dir, f"{prefix}_summary.txt")
    with open(summary_path, "w") as file: 
        file.write("Prediction Summary\n") 
        file.write("==================\n")
        file.write(f"Total 30-Second Patches: {total_patches}\n")

        #for loop to count how many patches were predicted for each class
        for i, class_name in enumerate(class_names): 
            class_count = (hard_preds == i).sum().item()
            if total_patches > 0: 
                percentage = (class_count / total_patches) * 100
            else:
                percentage = 0
            
            file.write(f"{class_name}: {class_count} ({percentage:.2f}%)\n")


soft_preds = torch.tensor([
    [0.70, 0.20, 0.05, 0.05],
    [0.10, 0.70, 0.10, 0.10],
    [0.05, 0.10, 0.80, 0.05],
    [0.60, 0.20, 0.10, 0.10]
])

output_dir = "."
prefix = "linear_sleepstages"

save_prediction_summary(soft_preds, output_dir, prefix)