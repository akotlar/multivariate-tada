#!/usr/bin/perl
use IO::Zlib;
use strict vars;
use Math::Random qw(:all);
use Math::Gauss ':all';

use vars qw(@fields @mom @dad @kid $npar);

if(@ARGV != 12) 
{
	print "\n Usage: ${0} Prev_Disorder1 Prev_Disorder2 Sample_Size No_Genes Mean_Rare_Freq_Per_Gene FractionGenes1_only FractionGenes2_only FractionBoth Rare_h2_1 rare_h2_2 rho outfile \n\n "; 
	exit(1);
}

open(FILE_SS,">$ARGV[11].ss") || die "\n Can not open $ARGV[11].ss for writing \n";

open(FILE_SS,">$ARGV[11].ss") || die "\n Can not open $ARGV[11].ss for writing \n";
print FILE_SS "Args are: " . "Prev_Disorder1,Prev_Disorder2,Sample_Size,No_Genes,Mean_Rare_Freq_Per_Gene,FractionGenes1_only,FractionGenes2_only,FractionBoth,Rare_h2_1,rare_h2_2,rho,outfile\n";
print FILE_SS "Args val: " . join(",", @ARGV) . "\n";

my @prev;
my @thres;

my @seed = random_get_seed();
random_set_seed(@seed);

for(my $i = 0 ; $i < 2; $i++)
{
	$prev[$i] = $ARGV[$i]+0;
	$thres[$i] = inv_cdf(1.0 - $prev[$i]);
	print FILE_SS "\n Disorder $i has a prev = $prev[$i] and thres = $thres[$i] \n";
}

my $Tot_N = $ARGV[2]+0;
my $Tot_G = $ARGV[3]+0;
my $Freq_P = $ARGV[4]+0;
my @model_p;
$model_p[1] = $ARGV[5]+0;
$model_p[2] = $ARGV[6]+0;
$model_p[3] = $ARGV[7]+0;
$model_p[0] = 1.0 - ($model_p[1] + $model_p[2] + $model_p[3]);

for(my $i =0; $i <4; $i++)
{
	if( ($model_p[$i] > 1) || ($model_p[$i] < 0.0))
	{
		die "\n Can not deal with a p = $model_p[$i] for fraction $i \n";
	}
}

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

print FILE_SS "\n h2[0] = $h2[0]  h2[1] =$h2[1]   rho = $rho \n";

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
$sigma[0] = 0.25;
$mu[0] = 1.0;
$mu[1] = 1.0;
$sigma[1] = 0.25;
my $nu = 0.0; 
if($rho < 0.9999999999)
{
	$nu = sqrt(1.0 - $rho*$rho) * $sigma[1]*$sigma[1];
}
my $lam = ($sigma[1] / $sigma[0]) * $rho;
my @stupid_sum;

# 0 = gene does not increase risk of disease
# 1 = gene increases risk for disease 2 but not disease 1
# 2 = gene increases risk for disease 2 but not disease 1
# 3 = gene increase risk for both disease 1 and disease 2
my @gene_architecture = ();
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
			push(@gene_architecture, 3);
			my $this_p = $this_c / (2.0*$Tot_N);
			my $this_q = 1.0 - $this_p;
			my @ttemp = random_normal(2,0,1);
			my @alpha0;
			my @alpha1;
			$alpha0[0] = $ttemp[0]*$sigma[0] + $mu[0];
			$alpha0[1] = ($alpha0[0] - $mu[0])*$lam + $mu[1] + $nu*$ttemp[1];
			for(my $m_hit = 0; $m_hit < 2; $m_hit++)
			{
				$alpha1[$m_hit] = -$this_p * $alpha0[$m_hit] / $this_q;  
				my $this_var = 2.0 * ($this_p * $alpha0[$m_hit]* $alpha0[$m_hit] + $this_q * $alpha1[$m_hit]*$alpha1[$m_hit]);
				$total_variance[$m_hit] += $this_var;
				my $a2 = 2.0 * $alpha1[$m_hit];
				for(my $j = 0; $j < $Tot_N;$j++)
				{
					$liability[$j][$m_hit] += $a2;
					$stupid_sum[$m_hit] += $a2;
					$this_stupid[$m_hit] += $a2;
				}
				my $adiff = $alpha0[$m_hit] - $alpha1[$m_hit];
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
				push(@gene_architecture, 1);
			} else {
				push(@gene_architecture, 2);
			}
			my $alpha0 = random_normal(1,$mu[$m_hit],$sigma[$m_hit]);
			my $this_p = $this_c / (2.0*$Tot_N);
			my $this_q = 1.0 - $this_p;
			my $alpha1 = -$this_p * $alpha0 / $this_q;
			my $this_var = 2.0 * ($this_p * $alpha0*$alpha0 + $this_q*$alpha1*$alpha1);
			$total_variance[$m_hit] += $this_var;
			my $a2 = 2.0 * $alpha1;
			for(my $j = 0; $j < $Tot_N;$j++)
			{
				$liability[$j][$m_hit] += $a2;
				$stupid_sum[$m_hit] += $a2;
				$this_stupid[$m_hit] += $a2;
			}
			my $adiff = $alpha0 - $alpha1;
			for(my $j = 1; $j <= $this_c; $j++)
			{
				$liability[$rare_allele_carriers[$i][$j]][$m_hit] += $adiff;
				$stupid_sum[$m_hit] += $adiff;	
				$this_stupid[$m_hit] += $adiff;
			}
		} else {
			push(@gene_architecture, 0);
		}
	} else {
		die("WTF count is 0");
	}
	#if((abs($this_stupid[0]) > 1e-15) || (abs($this_stupid[1]) > 1e-15))
	#{
	#	print "\n For gene $i Stupid_sum[0] = $this_stupid[0]   and stupid_sum[1] = $this_stupid[1] \n";
	#}
}

print FILE_SS "\n Stupid_sum[0] = $stupid_sum[0]   and stupid_sum[1] = $stupid_sum[1] \n";


my @affected;
my @res_var;
my @tot_affected;
my @gen_liab_mean;
my @res_liab_mean;
my @gen_liab_var;
my @res_liab_var;
my @tot_liab_mean;
my @tot_liab_var;

print FILE_SS "\n Before normalization total variances were $total_variance[0] and $total_variance[1] \n";


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
	$res_var[$j] = (1.0 - $h2[$j])*($total_variance[$j]/$h2[$j]);
	if($res_var[$j] > 1e-16)
	{
		my $res_sd = sqrt($res_var[$j]);
		for(my $i = 0; $i < $Tot_N;$i++)
		{
			my $this_e = random_normal(1,0,$res_sd);
			$res_liab_mean[$j] += $this_e;
			$res_liab_var[$j] += $this_e*$this_e;
			$liability[$i][$j] += $this_e;
		}
	}
	$gen_liab_mean[$j] /= $Tot_N;	
	$res_liab_mean[$j] /= $Tot_N;	
	$gen_liab_var[$j] /= $Tot_N;	
	$res_liab_var[$j] /= $Tot_N;	
	$gen_liab_var[$j] -= $gen_liab_mean[$j]*$gen_liab_mean[$j];	
	$res_liab_var[$j] -= $res_liab_mean[$j]*$res_liab_mean[$j];	

	my $norm_v = 1.0 / sqrt(($res_var[$j] + $total_variance[$j]));
	print FILE_SS "\nNormalizing by $norm_v\n";
	my $tot_mean = $gen_liab_mean[$j] + $res_liab_mean[$j];
	for(my $i = 0; $i < $Tot_N;$i++)
	{
		$liability[$i][$j] -= $tot_mean;
		$liability[$i][$j] *= $norm_v;
		$affected[$i][$j] = 0;
		$tot_liab_mean[$j] += $liability[$i][$j];
		$tot_liab_var[$j] += $liability[$i][$j]*$liability[$i][$j];
		if($liability[$i][$j] >= $thres[$j])
		{
			$affected[$i][$j] = 1;
			$tot_affected[$j]++;
		}
		if($j == 1)
		{
			if($affected[$i][0] && $affected[$i][1])
			{
				$both_affected++;
			}
		}
	}
	my $k = $tot_affected[$j] / $Tot_N;
	$o_prev[$j] = $k;
	$tot_liab_mean[$j] /= $Tot_N;	
	$tot_liab_var[$j] /= $Tot_N;	
	$tot_liab_var[$j] -= $tot_liab_mean[$j]*$tot_liab_mean[$j];	
	
	print FILE_SS "\n For disorder $j we expected a prevalence of $prev[$j] and got $k with $tot_affected[$j] out of $Tot_N\n";
	print FILE_SS "\n Genetic mean liability  = $gen_liab_mean[$j] Genetic Variance in Liabilty = $gen_liab_var[$j]"; 
	print FILE_SS "\n Residual mean liability  = $res_liab_mean[$j] Residual Variance in Liabilty = $res_liab_var[$j]"; 
	print FILE_SS "\n Total mean liability  = $tot_liab_mean[$j] Total Variance in Liabilty = $tot_liab_var[$j]\n"; 
}

$o_prev[2] = $both_affected / $Tot_N;

print FILE_SS "\n\nFinal Observed Prevalences for this study are (Disorder1,Disorder2,Both) = $o_prev[0],$o_prev[1],$o_prev[2]\n"; 
open(FILE,">$ARGV[11]") || die "\n Can not open $ARGV[11] for writing \n";
print FILE "Per_Gene_Counts_Unaffected_Unaffected,Unaffected_Affected,Affected_Unaffected,Affected_Affected,Gene_Architecture\n";


for(my $i = 0; $i < $Tot_G; $i++)
{
	my @aff_c;
	$aff_c[0][0] = $aff_c[0][1] = $aff_c[1][0] = $aff_c[1][1] = 0;
	for(my $j = 1; $j <= $rare_allele_carriers[$i][0]; $j++)
	{
		my $this = $rare_allele_carriers[$i][$j];
		$aff_c[$affected[$this][0]][$affected[$this][1]]++;
	}
	print FILE "$aff_c[0][0],$aff_c[0][1],$aff_c[1][0],$aff_c[1][1],$gene_architecture[$i]\n";
}
close(FILE);

