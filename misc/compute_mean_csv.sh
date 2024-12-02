calculate_means_all_csv() {
    for file in *.csv; do
        echo "Processing $file"
        awk -F, '
            NR > 1 {
                for (i = 1; i <= NF; i++) {
                    sum[i] += $i
                    count[i]++
                }
            }
            END {
                printf "File: %s\n", FILENAME
                for (i = 1; i <= length(sum); i++) {
                    printf "Column %d Mean: %.4f\n", i, sum[i] / count[i]
                }
                printf "\n"
            }
        ' "$file"
    done
}

calculate_means_all_csv
