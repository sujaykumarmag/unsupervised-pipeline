################################################################################################################

# Referenced from  : https://github.com/MariaNattestad/pca-on-genotypes/blob/main/vcf_to_matrix.py
# Notes : Converts the .vcf (variant cell format) to .csv file with the Population code 
#         of Genotypes(more alleles) file with the Chromosome Location.

################################################################################################################



from pysam import VariantFile
import numpy as np
import pandas as pd

vcf_filename = "datasets/genome/ALL.chr22.phase1_release_v3.20101123.snps_indels_svs.genotypes.vcf.gz"
panel_filename = "datasets/genome/phase1_integrated_calls.20101123.ALL.panel"

genotypes = []
samples = []
variant_ids = []
with VariantFile(vcf_filename) as vcf_reader:
    counter = 0
    for record in vcf_reader:
        counter += 1
        if counter % 100 == 0:
            alleles = [record.samples[x].allele_indices for x in record.samples]
            samples = [sample for sample in record.samples]
            genotypes.append(alleles)
            variant_ids.append(record.id)
        if counter % 4943 == 0:
            print(counter)
            print(f'{round(100 * counter / 494328)}%')


with open(panel_filename) as panel_file:
    labels = {}  # {sample_id: population_code}
    for line in panel_file:
        line = line.strip().split('\t')
        labels[line[0]] = line[1]


print(variant_ids)
genotypes = np.array(genotypes)
print(genotypes.shape)

matrix = np.count_nonzero(genotypes, axis=2)

matrix = matrix.T
print(matrix.shape)


df = pd.DataFrame(matrix, columns=variant_ids, index=samples)
df['Population code'] = df.index.map(labels)
df.to_csv("matrix.csv")