#!/usr/bin/perl
use IO::Zlib;
use strict vars;
use Math::Random qw(:all);
use Math::Gauss ':all';
use vars qw(@fields @mom @dad @kid $npar);
use JSON::XS qw(encode_json);
use DDP;
if(@ARGV != 12) 
{
	print "\n Usage: ${0} Prev_Disorder1 Prev_Disorder2 Sample_Size No_Genes Mean_Rare_Freq_Per_Gene FractionGenes1_only FractionGenes2_only FractionBoth Rare_h2_1 rare_h2_2 rho outfile\n\n"; 
	exit(1);
}

my %json;
my @prev;
my @thres;

#my @seed = random_get_seed();
#random_set_seed(@seed);
srand();

for(my $i = 0 ; $i < 2; $i++)
{
	$prev[$i] = $ARGV[$i]+0;
	$thres[$i] = inv_cdf(1.0 - $prev[$i]);
	print "\n Disorder $i has a prev = $prev[$i] and thres = $thres[$i] \n";
}

my $Tot_N = $ARGV[2]+0;
my $Tot_G = $ARGV[3]+0;
my $Freq_P = $ARGV[4]+0;
my @model_p;
$model_p[1] = $ARGV[5]+0;
$model_p[2] = $ARGV[6]+0;
$model_p[3] = $ARGV[7]+0;
$model_p[0] = 1.0 - ($model_p[1] + $model_p[2] + $model_p[3]);

open(cFILE,">$ARGV[11].cheat.tsv") || die "\n Can not open $ARGV[11].cheat.tsv for writing \n";
print cFILE "Gene_No\tModel_No\t[[a,b][c,d]]a\tb\tc\td\n";

for(my $i = 0; $i <4; $i++)
{
	if( ($model_p[$i] > 1) || ($model_p[$i] < 0.0))
	{
		die "\n Can not deal with a p = $model_p[$i] for fraction $i \n";
	}
}
print "\n Models: ($model_p[0],$model_p[1],$model_p[2],$model_p[3])\n";
$json{'models'} = [$model_p[0],$model_p[1],$model_p[2],$model_p[3]];
my @h2;

$h2[0] = $ARGV[8] + 0;
$h2[1] = $ARGV[9] + 0;

for(my $i =0; $i <2; $i++)
{
	if( ($h2[$i] > 1) || ($h2[$i] < 0.0))
	{
		die "\n Can not deal with a h2[$i]  = $h2[$i]; heritabilities must be between 0 and 1 \n";
	}
}
my $rho = $ARGV[10]+0;

print "\n h2[0] = $h2[0]  h2[1] =$h2[1]   rho = $rho \n";
$json{'h2'} = [$h2[0], $h2[1]];
$json{'rho'} = $rho;

if( (abs($rho) > 1))
{
	die "\n Can't deal with corr outside of -1 and 1.. rho = $rho\n";
} 

my @liability;
my @rare_allele_carriers;
my @total_variance;
my @sigma;
my @mu;

my $lambda = 2*$Tot_N * $Freq_P;
print('$lambda', $lambda);
$sigma[0] = sqrt($h2[0] / (2.0*$Freq_P*(1.0-$Freq_P)*($model_p[1]+$model_p[3])*$Tot_G));
$mu[0] = 0.0;
$mu[1] = 0.0;
$sigma[1] = sqrt($h2[1] / (2.0*$Freq_P*(1.0-$Freq_P)*($model_p[2]+$model_p[3])*$Tot_G));
my $nu = 0.0; 
my $jjj = ($lambda * $Tot_G)/$Tot_N;
my $kkk = $jjj * (1.0 - $model_p[0]);
print  "\n Using sigma[0] = $sigma[0] and sigma[1] = $sigma[1] and expected number of mutations per person = $jjj of which $kkk affect disease\n";
$json{'sigma'} = [$sigma[0], $sigma[1]];
$json{'expected_mutations_per_person'} = $jjj;
$json{'expected_risk_mutations_per_person'} = $kkk;

if($rho < 0.9999999999)
{
	$nu = sqrt((1.0 - $rho*$rho) * $sigma[1]*$sigma[1]);
}
my $lam = ($sigma[1] / $sigma[0]) * $rho;
my @stupid_sum;

my @d_as;

my @gene_architectures;
for(my $i = 0; $i < $Tot_G; $i++)
{
	# Choose the number of rare allele carriers $this_c and then who they are $rare_allele_carriers[$i][1..$this_c] 
	my $this_c = random_poisson(1,$lambda);
	$rare_allele_carriers[$i][0] = $this_c;
	# print "\n For i = $i we have lambda = $lambda  Totala count $this_c \n";
	my @this_stupid;
	$this_stupid[0] = 0.0;
	$this_stupid[1] = 0.0;
	if($this_c > 0)
	{
		my @temp = random_uniform_integer($this_c,0,$Tot_N-1);
		for(my $j = 1; $j <=$this_c;$j++)
		{
			$rare_allele_carriers[$i][$j] = $temp[$j-1];
			if( ($temp[$j-1] < 0) || ($temp[$j-1] >= $Tot_N))
			{
				die "\n This is not possible for j = $j-1 and temp = $temp[$j-1]\n";
			}
			# print "\n Found person $temp[$j-1]";
		}
		my $temp = random_uniform();
		if($temp > 1.0 - $model_p[3])
		{
			# Both diseases affected;
			print cFILE "$i\t3\t$ARGV[8]\t$ARGV[10]\t$ARGV[10]\t$ARGV[9]\n";
			push @gene_architectures, 3;
			my $this_p = $this_c / (2.0*$Tot_N);
			my $this_q = 1.0 - $this_p;
			my @ttemp = random_normal(2,0,1);
			my @beta;
			$beta[0] = ($ttemp[0]*$sigma[0] + $mu[0]); 	
			$beta[1] = ($beta[0] - $mu[0])*$lam + $mu[1] + $nu*$ttemp[1];
			if($beta[0] < 0)
			{
				if($beta[1] < 0)
				{
					$beta[0] = -$beta[0];
					$beta[1] = -$beta[1];
				}
				else
				{
					if(abs($beta[0]) > abs($beta[1]))
					{
						$beta[0] = -$beta[0];
						$beta[1] = -$beta[1];
					}
				}
			}
			elsif($beta[1] < 0)
			{
				if(abs($beta[1]) > abs($beta[0]))
				{
					$beta[0] = -$beta[0];
					$beta[1] = -$beta[1];
				}
			}
			# p @beta;
			my @alpha0;
			my @alpha1;
			$alpha0[0] = -$this_p*$beta[0];
			$alpha1[0] = $this_q*$beta[0];
			
			$alpha0[1] = -$this_p*$beta[1];
			$alpha1[1] = $this_q*$beta[1];
				
			$d_as[2]++;
			for(my $m_hit = 0; $m_hit < 2; $m_hit++)
			{
				my $this_var = 2.0 * ($this_p * $this_q * $beta[$m_hit]*$beta[$m_hit]);
				$total_variance[$m_hit] += $this_var;
				my $a2 = 2.0 * $alpha0[$m_hit];
				for(my $j = 0; $j < $Tot_N;$j++)
				{
					$liability[$j][$m_hit] += $a2;
					$stupid_sum[$m_hit] += $a2;
					$this_stupid[$m_hit] += $a2;
				}
				my $adiff = $beta[$m_hit];
				for(my $j = 1; $j <= $this_c; $j++)
				{
					$liability[$rare_allele_carriers[$i][$j]][$m_hit] += $adiff;
					$stupid_sum[$m_hit] += $adiff;	
					$this_stupid[$m_hit] += $adiff;
				}
			}
		
		}
		elsif($temp > $model_p[0])
		{
			my $m_hit = 0;
			if($temp > $model_p[0]+$model_p[1])
			{
				$m_hit = 1;
				print cFILE "$i\t1\t0\t0\t0\t$ARGV[9]\n";
				push @gene_architectures, 1;
			}
			else
			{
				print cFILE "$i\t2\t$ARGV[8]\t0\t0\t0\n";
				push @gene_architectures, 2;
			}
			#print "\n temp = $temp m_hit = $m_hit \n";
			$d_as[$m_hit]++;
			my $beta;
			my $alpha0;
			my $this_p = $this_c / (2.0*$Tot_N);
			my $this_q = 1.0 - $this_p;
			$beta = abs(random_normal(1,$mu[$m_hit],$sigma[$m_hit]));
			$alpha0 = -$this_p*$beta;
			my $alpha1 = $this_q * $beta;
			my $this_var = 2.0 * ($this_p * $this_q * $beta*$beta);
			$total_variance[$m_hit] += $this_var;
			my $a2 = 2.0 * $alpha0;
			for(my $j = 0; $j < $Tot_N;$j++)
			{
				$liability[$j][$m_hit] += $a2;
				$stupid_sum[$m_hit] += $a2;
				$this_stupid[$m_hit] += $a2;
			}
			my $adiff = $beta;
			for(my $j = 1; $j <= $this_c; $j++)
			{
				$liability[$rare_allele_carriers[$i][$j]][$m_hit] += $adiff;
				$stupid_sum[$m_hit] += $adiff;	
				$this_stupid[$m_hit] += $adiff;
			}
		}
		else
		{
			print cFILE "$i\t0\t0\t0\t0\t0\n";
			push @gene_architectures, 0;
		}
	}
#	if((abs($this_stupid[0]) > 1e-15) || (abs($this_stupid[1]) > 1e-15))
#	{
#		print "\n For gene $i Stupid_sum[0] = $this_stupid[0]   and stupid_sum[1] = $this_stupid[1] \n";
#	}
}
close(cFile);
print "\n Number of disease genes for 1 only $d_as[0], 2 only $d_as[1]  Both $d_as[2] \n";
$json{'disease_gene_counts'} = [int($d_as[0]), int($d_as[1]), int($d_as[2])];

print "\n Stupid_sum[0] = $stupid_sum[0]   and stupid_sum[1] = $stupid_sum[1] \n";
$json{'stupid_sum'} = [$stupid_sum[0], $stupid_sum[1]];

my @affected;
my @res_var;
my @tot_affected;
my @gen_liab_mean;
my @res_liab_mean;
my @gen_liab_var;
my @res_liab_var;
my @tot_liab_mean;
my @tot_liab_var;

print "\n Before normalization total variances were $total_variance[0] and $total_variance[1] \n";
$json{'total_variance_before_normalization'} = [$total_variance[0], $total_variance[1]];
$json{'total_N'} = $Tot_N;
$json{'prevalence_expected'} = [];
$json{'prevalence_actual'} = [];
$json{'total_affected'} = [];
$json{'genetic_mean_liability'} = [];
$json{'genetic_variance_liability'} = [];
$json{'residual_mean_liability'} = [];
$json{'residual_variance_liability'} = [];

#
#normalize phenotype and assign affectation status 
my $both_affected = 0;
my @o_prev;
for(my $j=0 ; $j < 2; $j++)
{
	for(my $i = 0; $i < $Tot_N;$i++)
	{
		$gen_liab_mean[$j] += $liability[$i][$j];
		$gen_liab_var[$j] += $liability[$i][$j]*$liability[$i][$j];
	}
	$res_var[$j] = 1.0 - $h2[$j];
	if($res_var[$j] > 1e-16)
	{
		my $res_sd = sqrt($res_var[$j]);
		for(my $i = 0; $i < $Tot_N;$i++)
		{
			my $this_e = random_normal(1,0,$res_sd);
			$res_liab_mean[$j] += $this_e;
			$res_liab_var[$j] += $this_e*$this_e;
			$liability[$i][$j] += $this_e;
			if($liability[$i][$j] >= $thres[$j])
			{
				$affected[$i][$j] = 1;
				$tot_affected[$j]++;
			}
		}
	}
	$gen_liab_mean[$j] /= $Tot_N;	
	$res_liab_mean[$j] /= $Tot_N;	
	$gen_liab_var[$j] /= $Tot_N;	
	$res_liab_var[$j] /= $Tot_N;	
	$gen_liab_var[$j] -= $gen_liab_mean[$j]*$gen_liab_mean[$j];	
	$res_liab_var[$j] -= $res_liab_mean[$j]*$res_liab_mean[$j];	

	my $k = $tot_affected[$j] / $Tot_N;
	$o_prev[$j] = $k;
	
	print "\n For disorder $j we expected a prevalence of $prev[$j] and got $k with $tot_affected[$j] out of $Tot_N\n";
	print "\n Genetic mean liability  = $gen_liab_mean[$j] Genetic Variance in Liabilty = $gen_liab_var[$j]"; 
	print "\n Residual mean liability  = $res_liab_mean[$j] Residual Variance in Liabilty = $res_liab_var[$j]"; 


	push @{$json{'prevalence_expected'}}, $prev[$j];
	push @{$json{'prevalence_actual'}}, $k;
	push @{$json{'total_affected'}}, int($tot_affected[$j]);
	push @{$json{'genetic_mean_liability'}}, $gen_liab_mean[$j];
	push @{$json{'genetic_variance_liability'}}, $gen_liab_var[$j];
	push @{$json{'residual_mean_liability'}}, $res_liab_mean[$j];
	push @{$json{'residual_variance_liability'}}, $res_liab_var[$j];
}
for(my $i = 0; $i < $Tot_N;$i++)
{
	if($affected[$i][0] && $affected[$i][1])
	{
		$both_affected++;
	}	
}

$o_prev[2] = $both_affected / $Tot_N;

print "\n\nFinal Observed Prevalences for this study are (Disorder1,Disorder2,Both) = $o_prev[0],$o_prev[1],$o_prev[2]\n";
$json{'observed_prevalences'} = [$o_prev[0],$o_prev[1],$o_prev[2]];

open(FILE,">$ARGV[11]") || die "\n Can not open $ARGV[11] for writing \n";
open(C2FILE,">$ARGV[11].cheat2.csv") || die "\n Can not open $ARGV[11].cheat2.csv for writing \n";
print FILE "Per_Gene_Counts_Unaffected_Unaffected,Unaffected_Affected,Affected_Unaffected,Affected_Affected\n";
print C2FILE "Per_Gene_Counts_Unaffected_Unaffected,Unaffected_Affected,Affected_Unaffected,Affected_Affected,Gene_Model\n";

for(my $i = 0; $i < $Tot_G; $i++)
{
	my @aff_c;
	$aff_c[0][0] = $aff_c[0][1] = $aff_c[1][0] = $aff_c[1][1] = 0;
	for(my $j = 1; $j <= $rare_allele_carriers[$i][0]; $j++)
	{
		my $this = $rare_allele_carriers[$i][$j];
		$aff_c[$affected[$this][0]][$affected[$this][1]]++;
	}
	print FILE "$aff_c[0][0],$aff_c[0][1],$aff_c[1][0],$aff_c[1][1]\n";
	print C2FILE "$aff_c[0][0],$aff_c[0][1],$aff_c[1][0],$aff_c[1][1],$gene_architectures[$i]\n";
}
p %json;
my $json = encode_json (\%json);

print($json);
open(JFILE,">$ARGV[11].json") || die "\n Can not open $ARGV[11].json for writing \n";
print JFILE $json;

close(FILE);


