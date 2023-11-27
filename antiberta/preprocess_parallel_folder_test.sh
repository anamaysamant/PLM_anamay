#!/bin/bash
module load eth_proxy

echo "Input file: $1"
echo "Output file: $2"
echo "Every n-th line: $3"

#get length of the input file
length=$(wc -l < "$1")
echo "Length of the input file: $length"

#calculate the number of lines in each file
lines_per_file=$(( (length + 49) / 50 ))
echo "Number of lines in each file: $lines_per_file"

#split the input file into 50 files
split -l $lines_per_file "$1" output

declare -a job_ids

# iterate over the 50 files and run the preprocess script on each of them
for file in output*; do
    echo "Processing file $file"
    
    # create a new directory for each subfile
    dir_name="${file}_dir"
    mkdir "$dir_name"
    
    # copy the subfile to its directory
    mv "$file" "$dir_name"
    
    # set the output_file name
    output_file="${file}_out"
    
    # submit separate jobs for each of the 50 files
    job_id=$(sbatch -A es_reddy -n 1 --cpus-per-task=1 --time=24:00:00 --mem-per-cpu=20048 --job-name="download $file" --wrap="cd $dir_name && bash ../../../preprocess.sh $file $output_file $3" | awk '{print $4}')
    job_ids+=($job_id)
done

# make sure the output file is empty
> "$2.txt"

for id in "${job_ids[@]}"; do
    while squeue -j $id | grep -q $id; do
        echo "Waiting for job $id to finish..."
        sleep 60
    done
done

# combine the output files
for dir in output*_dir; do
    file="${dir}/${dir/_dir/_out}.txt"
    echo "Combining file $file"
    cat "$file" >> "$2.txt"
done


# remove the intermediate output files
# rm output*

echo "Data has been written to $2.txt."
