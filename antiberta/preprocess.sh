#!/bin/bash
#downloading the files from OAS
echo "Input file: $1"
echo "Output file: $2"
echo "Every n-th line: $3"

output_file="${2}.txt"
# Clear the output file
> "$output_file"

column_name="sequence_alignment_aa"


count=1
while read -r line; do 
    if (($count % $3 == 0)); then
        eval $line  
        # unzipping the files
        files=./*
        echo "Unzipping files"
        for file in $files
        do
            #echo $(basename $file)
            if [ "${file:${#file}-2:${#file}}" == "gz" ]; then
            gunzip -d $(basename $file)
            fi
        done
        for file in *.csv; do
        # Check if the file exists and is readable
        if [ -r "$file" ]; then
                local column
                unset column
            # Extract the desired column and append it to the output file
                column=$(awk -F "\"*,\"*" 'NR>2 && length($35) >= 20 && length($45)  >= 10 && length($37) >= 5 && length($37) <= 12 && length($41) <= 10 && length($41) >= 1 && length($47) >= 5 && length($47) <= 38 {print $14}' "$file")
                if [ -n "$column" ]; then
                    printf "%s\n" "$column" >> "$output_file"
                fi            #printf "%s\n" $column > $output_file
        fi
        done
        rm -f *.csv
        rm -f *.gz  
    fi
    ((count++))
done < $1

echo "Data has been written to $output_file."

# unzipping the files (this is now done after every file)
#files=./*
#for file in $files
#do
 #echo $(basename $file)
 #if [ "${file:${#file}-2:${#file}}" == "gz" ]; then
  #gunzip -d $(basename $file)
 #fi
#done
#!/bin/bash


# Iterate over all CSV files in the current directory (this is now done after every file)
#for file in *.csv; do
    # Check if the file exists and is readable
    #if [ -r "$file" ]; then
        # Extract the desired column and append it to the output file
        #column=$(awk -F "\"*,\"*" 'NR>2 {print $13}' "$file")
        #printf "%s\n" $column > $output_file
    #fi
#done


