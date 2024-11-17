import random
import sys

if __name__ == "__main__":


    label_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(label_file, 'r') as f:
        lines = f.readlines()
    lines = lines[1:]  # 去掉第一行

    list_all = []
    mapping = {}
    mapping_count = 0
    instance_count = 0

    with open(output_file, 'w') as f:
        for line in lines:
            parts = line.strip().split(',')
            label = parts[0]

            if label not in mapping:
                mapping[label] = mapping_count
                mapping_count += 1

            img_list = parts[1].split()
            
            for img in img_list:
                img_path = f"{img[0]}/{img[1]}/{img[2]}/{img}.jpg"
                list_all.append(
                    f"{mapping[label]}\t{img_path}"
                )

        random.shuffle(list_all)
        for line in list_all:
            f.write(f"{instance_count}\t{line}" + '\n')
            instance_count += 1
