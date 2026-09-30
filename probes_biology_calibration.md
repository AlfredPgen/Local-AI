# Difficulty calibration of probes_biology.tsv

2026-09-29 23:35; 336 items; labels: easy if at least 3/4 of the reference models are right, hard if at most 1/4, medium otherwise.

| Reference model | Parameters | Accuracy | Floor | Sanity loss |
|---|---|---|---|---|
| HuggingFaceTB/SmolLM2-135M | 135M | 38.1% | - | 2.52 |
| HuggingFaceTB/SmolLM2-360M | 362M | 51.5% | - | 2.20 |
| TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T | 1,100M | 42.9% | - | 2.40 |
| HuggingFaceTB/SmolLM2-1.7B | 1,711M | 67.6% | - | 2.09 |

Labels: easy 144, medium 44, hard 148.

## Accuracy by category

| Category | SmolLM2-135M | SmolLM2-360M | TinyLlama-1.1B-intermediate-step-1431k-3T | SmolLM2-1.7B |
|---|---|---|---|---|
| biochemistry and proteins | 30% | 45% | 40% | 60% |
| cell biology | 60% | 60% | 60% | 80% |
| databases and biobanks | 40% | 60% | 65% | 80% |
| evolution | 40% | 50% | 45% | 60% |
| genetics | 55% | 55% | 50% | 85% |
| histology and anatomy | 45% | 70% | 50% | 80% |
| history of science | 35% | 50% | 35% | 60% |
| machine learning | 35% | 50% | 45% | 75% |
| mathematics | 25% | 55% | 35% | 85% |
| medicine and physiology | 35% | 50% | 45% | 65% |
| molecular biology | 40% | 50% | 45% | 70% |
| pharmacology and drug discovery | 40% | 55% | 40% | 80% |
| population genetics | 33% | 42% | 38% | 38% |
| quantitative genetics | 12% | 17% | 8% | 25% |
| statistical genetics | 42% | 54% | 46% | 71% |
| statistics and probability | 46% | 67% | 46% | 79% |

## Items every reference model got wrong, all choosing the same distractor (48)

Check each: the distractor may also be correct, or the wording ambiguous.

- [population genetics] Under complete self-fertilisation, the proportion of heterozygous individuals, compared with the previous generation, is **halved** (all chose: *unchanged*; mean margin -0.27)
- [population genetics] In an idealised diploid population of 50 individuals, drift reduces expected heterozygosity each generation by **1 percent** (all chose: *10 percent*; mean margin -0.11)
- [population genetics] Using Wright's formula for unequal numbers of the sexes, a population with 10 breeding males and 90 breeding females has an effective size of **36** (all chose: *100*; mean margin -1.63)
- [population genetics] In a large population with a constant selfing rate of 50 percent, the equilibrium inbreeding coefficient is **1/3** (all chose: *1/2*; mean margin -0.59)
- [quantitative genetics] Selection on a trait with narrow-sense heritability 0.4 gives a selection differential of 30, so the expected response to selection is **12** (all chose: *30*; mean margin -1.52)
- [quantitative genetics] If environmental variance increases while genetic variance stays the same, the heritability of the trait **decreases** (all chose: *increases*; mean margin -0.14)
- [quantitative genetics] Falconer's liability-threshold model assumes that an unobserved liability to disease in the population follows a **normal distribution** (all chose: *Poisson distribution*; mean margin -0.04)
- [quantitative genetics] Because parents transmit alleles rather than whole genotypes, resemblance between parents and offspring depends mainly on the **additive variance** (all chose: *environmental variance*; mean margin -0.07)
- [quantitative genetics] Ignoring shared environment, the regression slope of offspring phenotype on the phenotype of a single parent estimates **half the heritability** (all chose: *the genetic correlation*; mean margin -0.32)
- [quantitative genetics] The phenotypic correlation between identical twins reared apart in uncorrelated environments directly estimates **broad-sense heritability** (all chose: *genetic correlation*; mean margin -0.29)
- [quantitative genetics] Under a dominance model of heterosis, the advantage of an F1 hybrid over its parents is reduced in the F2 generation by **one half** (all chose: *three quarters*; mean margin -0.18)
- [quantitative genetics] Under Fisher's infinitesimal model, selection shifts the trait mean while the frequency of each individual allele changes **negligibly** (all chose: *exponentially*; mean margin -0.26)
- [quantitative genetics] Inbreeding depression in the mean of a trait occurs only if the loci affecting that trait show directional **dominance** (all chose: *selection*; mean margin -0.43)
- [quantitative genetics] When the top half of a normally distributed population is selected, the standardised selection intensity is about **0.8** (all chose: *0.5*; mean margin -0.24)
- [evolution] According to the island rule, large mammals such as elephants that become isolated on small islands tend to evolve **dwarfism** (all chose: *flightlessness*; mean margin -0.37)
- [evolution] Under strict neutrality, if effective population size were ten times larger, the long-term rate of substitution would **stay the same** (all chose: *roughly double*; mean margin -0.12)
- [evolution] In haplodiploid ants and bees, full sisters sharing the same father have a coefficient of relatedness of **3/4** (all chose: *1/2*; mean margin -0.89)
- [evolution] Incomplete lineage sorting is most likely when speciation events are closely spaced and ancestral effective population sizes are **large** (all chose: *small*; mean margin -0.13)
- [evolution] In Fisher's geometric model of adaptation, a random mutation of vanishingly small size has a probability of being beneficial of **one half** (all chose: *one quarter*; mean margin -0.07)
- [history of science] In the Hershey-Chase experiment, the protein coats of bacteriophage T2 were labelled with radioactive **sulfur** (all chose: *phosphorus*; mean margin -0.42)
- [history of science] The first genetic linkage map, ordering genes along the Drosophila X chromosome, was constructed in 1913 by **Alfred Sturtevant** (all chose: *Thomas Hunt Morgan*; mean margin -0.18)
- [history of science] Walter Sutton developed the chromosome theory of inheritance from his observations of meiosis in the testes of **grasshoppers** (all chose: *fruit flies*; mean margin -0.12)
- [history of science] Crick and Brenner showed that the genetic code is read in triplets by combining frameshift mutations induced by **acridine dyes** (all chose: *ultraviolet light*; mean margin -0.67)
- [history of science] A classic long-term selection experiment for oil and protein content in maize began at the University of Illinois in **1896** (all chose: *1926*; mean margin -0.68)
- [history of science] Path analysis, a method for partitioning correlations along a diagram of causal links, was devised around 1920 by **Sewall Wright** (all chose: *Karl Pearson*; mean margin -0.20)
- [statistical genetics] The genomic inflation factor equals the median observed one-degree-of-freedom chi-squared association statistic divided by approximately **0.455** (all chose: *0.5*; mean margin -0.60)
- [statistical genetics] When LD score regression is applied to GWAS statistics from a study with no confounding or cryptic relatedness, the expected intercept is close to **1.0** (all chose: *0.0*; mean margin -0.23)
- [statistical genetics] In Mendelian randomization, a genetic instrument that affects the outcome through a pathway bypassing the exposure displays **horizontal pleiotropy** (all chose: *reverse causation*; mean margin -0.21)
- [statistical genetics] In the widely used coloc Bayesian method, the hypothesis that two traits share a single causal variant in the region is labelled **H4** (all chose: *H1*; mean margin -2.49)
- [statistics and probability] If 20 independent tests are each performed at the 0.05 level and every null hypothesis is true, the expected number of false positives is **one** (all chose: *zero*; mean margin -0.43)
- [statistics and probability] With 20 independent tests each at the 0.05 level and all null hypotheses true, the probability of at least one false positive is about **0.64** (all chose: *0.05*; mean margin -0.99)
- [statistics and probability] A test with 99% sensitivity and 99% specificity, applied to a disease with prevalence one in ten thousand, has a positive predictive value of about **1%** (all chose: *99%*; mean margin -1.11)
- [statistics and probability] A variable that is caused by both the exposure and the outcome, and that creates a spurious association when conditioned on, is called **a collider** (all chose: *a confounder*; mean margin -0.87)
- [statistics and probability] The lower bound that Chebyshev's inequality places on the proportion of any finite-variance distribution lying within two standard deviations of its mean is **75%** (all chose: *50%*; mean margin -0.58)
- [mathematics] Multiplying every entry of a 3 by 3 matrix by 2 multiplies its determinant by **8** (all chose: *2*; mean margin -2.23)
- [machine learning] He initialisation, designed for networks with ReLU activations, draws weights with a variance equal to **two over the fan-in** (all chose: *one over the fan-in*; mean margin -0.34)
- [machine learning] In the original dropout scheme, units are all kept at test time and their outgoing weights are multiplied by the **retention probability** (all chose: *dropout probability*; mean margin -0.38)
- [genetics] In a population in Hardy-Weinberg equilibrium where the recessive allele has frequency 0.1, the expected frequency of heterozygotes is **0.18** (all chose: *0.01*; mean margin -0.71)
- [molecular biology] In E. coli promoters recognised by sigma 70, the consensus sequence TTGACA is centred near position **-35** (all chose: *-10*; mean margin -0.40)
- [cell biology] Most proteins imported into the peroxisomal matrix carry a short C-terminal targeting signal with the sequence **SKL** (all chose: *KDEL*; mean margin -2.01)
- [medicine and physiology] On an electrocardiogram, repolarisation of the ventricles produces the **T wave** (all chose: *QRS complex*; mean margin -0.40)
- [medicine and physiology] The hormone that acts on principal cells of the collecting duct to increase sodium reabsorption and potassium secretion is **aldosterone** (all chose: *parathyroid hormone*; mean margin -0.12)
- [medicine and physiology] Hepcidin, the liver hormone that controls iron balance, lowers plasma iron by triggering internalisation and degradation of **ferroportin** (all chose: *transferrin*; mean margin -0.25)
- [medicine and physiology] In primary hyperaldosteronism caused by an adrenal adenoma, plasma renin activity is typically **suppressed** (all chose: *elevated*; mean margin -0.22)
- [histology and anatomy] At the base of the small intestinal crypts, the cells that secrete lysozyme and defensins are **Paneth cells** (all chose: *enteroendocrine cells*; mean margin -0.62)
- [pharmacology and drug discovery] Preclinical safety pharmacology screens drug candidates for QT prolongation risk by measuring block of the potassium channel encoded by **KCNH2** (all chose: *KCNQ1*; mean margin -0.55)
- [databases and biobanks] gnomAD, which aggregates exome and genome sequencing data from many studies, is mainly consulted to obtain a variant's **population allele frequency** (all chose: *clinical significance*; mean margin -0.11)
- [databases and biobanks] The China Kadoorie Biobank recruited about half a million adults from **ten regions** (all chose: *three regions*; mean margin -0.26)
