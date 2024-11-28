## export bin
export PATH=$PATH:/Users/datn/github/nf-deepsea/bin


cd /Users/datn/DATA_ANALYSES/OCR_prediction/cattle_test

bed_path=genome_window.bed


## slipt genome

generate_coordinate_onebed.py --genome /Users/datn/GENOMES/cattle/Bos_taurus.ARS-UCD1.2.dna.toplevel.fa.fai --out $bed_path --window 200 --chrom 29

## bed mapping



mkdir bed_mapping
for peak in /Users/datn/DATA_ANALYSES/OCR_prediction/data_downloaded/cattle/*
do
    out_fn=$(basename $peak)
    bedtools intersect -a $bed_path -b ${peak} -wo -f 0.50 > bed_mapping/${out_fn} &
done


generate_seq_labels_base_ratio.py --input bed_mapping/* --out 'label'


mkdir tfr_data
for i in {1..29}
do
    generate_tfr_fw.py --label label/${i}.txt.gz --bed genome_window.bed --out tfr_data/${i} --genome /Users/datn/GENOMES/cattle/Bos_taurus.ARS-UCD1.2.dna.toplevel.fa --pad_scale 5
done









########### SED-LD analysis
cd /Users/datn/DATA_ANALYSES/cattleGTEX/genotype

echo '##fileformat=VCFv4.1' > header.vcf
echo '##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">' >> header.vcf
echo '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">' >> header.vcf

gunzip cattle_Holstein.vcf.gz

cat header.vcf cattle_Holstein.vcf | bgzip > cattle_Holstein.vcf.gz

bcftools index -t cattle_Holstein.vcf.gz

bcftools view -r 1 cattle_Holstein.vcf.gz | bcftools annotate --set-id '%CHROM\_%POS\_%REF\_%ALT' | bgzip > cattle_Holstein_1.vcf.gz

#bcftools annotate --set-id '%CHROM\_%POS\_%REF\_%ALT' cattle_Holstein_1.vcf.gz | less


mkdir plink
plink --vcf cattle_Holstein_1.vcf.gz \
      --vcf-half-call 'haploid' \
      --make-bed  --const-fid --out ./plink/chr1 \
      --threads 1 \
      --memory 2000

plink --bfile ./plink/chr1 \
      --r --ld-window-r2 0.2 \
      --ld-window-kb 500 \
      --ld-window 5000 \
      --out ./plink/chr1_ld \
      --threads 8 \
      --memory 2000

mv ./plink/chr1_ld.ld ./



# predict variant effect vcf 

predict_impact_vcf.py --vcf /Users/datn/DATA_ANALYSES/cattleGTEX/genotype/cattle_Holstein_1.vcf.gz \
     --genome /Users/datn/GENOMES/cattle/Bos_taurus.ARS-UCD1.2.dna.toplevel.fa \
     --cols /Users/datn/DATA_ANALYSES/OCR_prediction/data_downloaded/cattle_colnames.txt \
     --model /Users/datn/DATA_ANALYSES/OCR_prediction/orion_trained/cattle_fw/danq_model.h5 --out impact_score_chrom1.txt