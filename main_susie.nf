#!/usr/bin/env nextflow
/*
========================================================================================
                          variant-impact-inference
========================================================================================
                          Author: Dat T Nguyen
                          Contact: ndat<at>utexas.edu
----------------------------------------------------------------------------------------
*/


/*
 Define the default parameters
*/ 

params.genome           = "$baseDir/data/refs/genome.fa"
params.vcfs             = "$baseDir/data/*vcf.gz"
params.col_file         = "$baseDir/data/col_files.txt"
params.model            = "$baseDir/data/model.h5"


params.outdir           = "susie_results"
params.trace_dir        = "susie_trace_dir"


nextflow.enable.dsl=2

workflow {

    VCF_ch = channel.fromPath(params.vcfs, checkIfExists: true)
    IPACT_inference(VCF_ch, params.genome, params.col_file, params.model)

}


process IPACT_inference {
    container 'ndatth/deepsea:v0.0.0'
    publishDir "${params.outdir}", mode: 'copy', overwrite: true
    memory '60 GB'
    cpus 8
    label 'with_1gpu'

    input:
    path vcf_file
    path genome
    path col_file
    path model

    output:
    path "*_impact_scores.txt.gz"


    script:
    """

    base_name=\$(basename ${vcf_file} .vcf.gz)

    predict_impact_vcf.py \
        --model ${model} \
        --vcf ${vcf_file} \
        --genome ${genome} \
        --model_file ${model_weight} \
        --cols ${col_file} \
        --out \${base_name}_impact_scores.txt \

    gzip \${base_name}_impact_scores.txt

    """
}



// process LD_compute {
//     container 'ndatth/pytorch:v0.0.0'
//     publishDir "${params.outdir}/${params.sub_trace_dir}", mode: 'link', overwrite: true
//     memory '16 GB'
//     cpus 2

//     input:
//     path vcf_file

//     output:
//     path "*.ld"


//     script:
//     """

//     base_name=\$(basename ${vcf_file} .vcf.gz)

//     mkdir -p tem_dir

//     plink --vcf $vcf_file --chr-set 30 no-xy --make-bed --out tem_dir/tem_

//     plink --bfile tem_dir/tem_ \
//         --chr-set 30 no-xy \
//         --r2 --ld-window-kb 100 \
//         --ld-window 200 \
//         --ld-window-r2 0 \
//         --memory 16000 \
//         --out out_fn

//     rm -rf tem_dir

//     mv out_fn.ld \${base_name}.ld

//     rm -rf tem_dir
//     """
// }




// process RExLD_all_string {
//     container 'ndatth/pytorch:v0.0.0'
//     publishDir "${params.outdir}/${params.sub_outdir}/RExLD_scores", mode: 'copy', overwrite: true
//     memory '8 GB'
//     cpus 1

//     input:

//     tuple path(impact_file), val(pick_string)
//     path ld_file_list

//     output:
//     path "*_reld_score.txt"


//     script:
//     """

//     base_name=\$(basename ${impact_file} _impact_scores.txt.gz)

//     cli_reld.py \
//         --score_file ${impact_file} \
//         --ld_file \${base_name}.ld \
//         --pick_string ${pick_string} \
//         --out_reld \${base_name}_${pick_string}_reld_score.txt

//     """
// }

