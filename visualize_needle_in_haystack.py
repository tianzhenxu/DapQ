
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import json
import glob

# FOLDER_PATH = "/opt/data/private/sy/tzx/idea_needle_in_haystack_result/llama/128/results_needle/results/LlaMA3_snapkv/"
MODEL_NAME = None
PRETRAINED_LEN=33000


def main(FOLDER_PATH):
    # Path to the directory containing JSON results
    folder_path = FOLDER_PATH
    kv_size=folder_path.split("/")[-5]
    if("/" in folder_path):
        model_name = folder_path.split("/")[-2]
    else: model_name = MODEL_NAME
    print("model_name = %s" % model_name)

    # Using glob to find all json files in the directory
    json_files = glob.glob(f"{folder_path}*.json")
    # import ipdb; ipdb.set_trace()

    # List to hold the data
    data = []

    # Iterating through each file and extract the 3 columns we need
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            # Extracting the required fields
            
            try:
                document_depth = json_data.get("depth_percent", None)
                context_length = json_data.get("context_length", None)
            except:
                import pdb
                pdb.set_trace()
            # score = json_data.get("score", None)
            model_response = json_data.get("model_response", None).lower()
            needle = json_data.get("needle", None).lower()
            expected_answer = "eat a sandwich and sit in Dolores Park on a sunny day.".lower().split()
            score = len(set(model_response.split()).intersection(set(expected_answer))) / len(set(expected_answer))
            # Appending to the list
            data.append({
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score
            })

    # Creating a DataFrame
    df = pd.DataFrame(data)

    locations = list(df["Context Length"].unique())
    locations.sort()
    for li, l in enumerate(locations):
        if(l > PRETRAINED_LEN): break
    pretrained_len = li

    print(df.head())
    print("Overall score %.3f" % df["Score"].mean())

    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    pivot_table.iloc[:5, :5]

    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    f = plt.figure(figsize=(30, 12))  # Can adjust these dimensions as needed
    heatmap = sns.heatmap(
        pivot_table,
        vmin=0, vmax=1,
        cmap=cmap,
        cbar_kws={
            'label': 'Score',
            'pad': 0.01,  # 控制颜色条与热力图的水平间距（默认0.05，越小越近，0.01为推荐值）
            # 不设置shrink，保持颜色条原始大小不变
        },
        linewidths=0.5,  # Adjust the thickness of the grid lines here
        linecolor='grey',  # Set the color of the grid lines
        linestyle='--',
        
    )
    cbar = heatmap.collections[0].colorbar
    cbar.set_label('Score', fontsize=50)
    cbar.ax.tick_params(labelsize=35)
    # More aesthetics
    
    if 'h2o' in model_name.lower():
        plt.title(f'H2O, KV Size={kv_size}, Acc={df["Score"].mean() * 100:.2f} ', fontsize=50)
    elif 'snapkv' in model_name.lower():
        plt.title(f'SnapKV, KV Size={kv_size}, Acc={df["Score"].mean() * 100:.2f} ', fontsize=50)
    elif 'pyramidkv' in model_name.lower():
        plt.title(f'PyramidKV, KV Size={kv_size}, Acc={df["Score"].mean() * 100:.2f} ', fontsize=50)
    elif 'ours' in model_name.lower():
        plt.title(f'Ours, KV Size={kv_size}, Acc={df["Score"].mean() * 100:.2f} ', fontsize=50)
    else:
        plt.title(f'FullKV, KV Size={kv_size}, Acc={df["Score"].mean() * 100:.2f} ', fontsize=50)
        
        
        
    plt.xlabel('Token Limit', fontsize=50)  # X-axis label
    plt.ylabel('Depth Percent', fontsize=50)  # Y-axis label
    
    # 修改横坐标标签显示：每隔两个显示一次
    ax = plt.gca()
    xticks = ax.get_xticks()
    xticklabels = [label.get_text() for label in ax.get_xticklabels()]
    
    # 创建新的标签列表和刻度位置列表，每隔两个显示一个
    new_xticklabels = []
    keep_ticks = []  # 保存需要保留刻度线的位置
    for i, label in enumerate(xticklabels):
        if i % 3 == 0:  # 每隔两个显示一个（索引0, 3, 6...）
            new_xticklabels.append(label)
            keep_ticks.append(xticks[i])

    # 设置新的横坐标标签和刻度线
    ax.set_xticks(keep_ticks)  # 只保留有标签位置的刻度线
    
    ax.set_xticklabels(new_xticklabels, rotation=45, fontsize=40)    
    # plt.xticks(rotation=45, fontsize=30)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0, fontsize=40)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    # Add a vertical line at the desired column index
    plt.axvline(x=pretrained_len + 0.8, color='white', linestyle='--', linewidth=4)

    save_path = "./results_needle/img/%s.pdf" % model_name
    print("saving at %s" % save_path)
    plt.savefig(save_path, dpi=600,bbox_inches='tight' )


if __name__ == "__main__":
    main("/opt/data/private/sy/tzx/idea_needle_in_haystack_result/llama/256/results_needle/results/LlaMA3_fp16/")
    main("/opt/data/private/sy/tzx/idea_needle_in_haystack_result/llama/256/results_needle/results/LlaMA3_h2o/")
    main("/opt/data/private/sy/tzx/idea_needle_in_haystack_result/llama/256/results_needle/results/LlaMA3_ours/")
    main("/opt/data/private/sy/tzx/idea_needle_in_haystack_result/llama/256/results_needle/results/LlaMA3_pyramidkv/")
    main("/opt/data/private/sy/tzx/idea_needle_in_haystack_result/llama/256/results_needle/results/LlaMA3_snapkv/")
