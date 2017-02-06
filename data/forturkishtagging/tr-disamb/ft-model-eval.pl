#!/usr/bin/perl -w
use strict;
use Data::Dumper;
use Getopt::Std;

my $modeldir = "ft-models.out";
my %opt; getopts('vsp', \%opt);
# -v for verbose
# -s for including stem probabilities
# -p for only including positive features in the product

warn "Reading err files for accuracy information\n";
my $acc = {};
for my $f (glob "$modeldir/*.err") {
    $f =~ /\/(.+?)\.err$/ or die "Bad filename $f";
    my $feature = $1;
    open(FP, $f) or die "Cannot open file $f";
    while(<FP>) { last if /TESTING/; }
    while(<FP>) { last if /percent/; }
    $_ = <FP>;
    if (/\t(\d+\.\d+|nan)\%$/) {
	$acc->{$feature} = ($1 eq 'nan' ? 100 : $1);
	warn "$acc->{$feature} accuracy for $feature\n" if $opt{v};
    } else {
	$acc->{$feature} = 100;
	warn "WARNING: Cannot get accuracy for $feature, using $acc->{$feature}";
    }
    close(FP);
}

warn "Loading models\n";
my $model = {};
for my $f (glob "$modeldir/*.out") {
    $f =~ /\/(.+?)\.out$/ or die "Bad filename $f";
    my $feature = $1;
    open(FP, $f) or die "Cannot open file $f";
    @{$model->{$feature}} = <FP>;
    close(FP);
    my $n = @{$model->{$feature}};
    if ($n == 0) {
	push @{$model->{$feature}}, '0';
	warn "WARNING: Empty model for $feature, using [0]";
    } else {
	warn "$n rules loaded for $feature\n" if $opt{v};
    }
}

my $freq = {};
if ($opt{s}) {
    warn "Loading stem frequencies\n";
    open(FP, "stem-count.out") or die $!;
    while(<FP>) {
	my ($n, $s) = split;
	$freq->{$s} = $n;
	$freq->{_ALL_} += $n;
    }
    close(FP);
}

warn "Disambiguating data\n";
my $stats;
while(<>) {
#    print STDERR;
    print STDERR "." if not $opt{v} and $.%100 == 0;
    my ($answer, $attr, $tags) = /^(\S+) ([^\#]+?) \# (.+)$/;
    die "Bad line [$_]" unless defined $tags;
    $stats->{total}++;
    my %attr;
    $attr{$_} = 1 for split(/\s+/, $attr);
    my @tags = split(/\s+/, $tags);
    
    # Figure out the word for debugging
    die "No word [$_][$attr]" unless ($attr =~ /\bW==(\S+)/);
    my $word = $1;
    warn "==> Disambiguating $word\n" if $opt{v};

    # If unambiguous just count as correct
    if (@tags == 1) {
	$stats->{unamb}++;
	die "Answer mismatch: [$_]" unless $answer eq $tags[0];
	$stats->{correct}++;
	warn "Found unambiguous: $word $answer\n" if $opt{v};
	print "$word $answer\n";
	next;
    }

    # Calculate the P(ft=1|dl) for each feature in all the tags
    my %prob;
    for my $tag (@tags) {
	$tag =~ /^(.+?)\+((\w+).*)$/ or die "Bad tag [$tag]";
	my $stem = "$1+$3";
	my $mtag = $2;
 	if ($opt{s} and not defined $prob{$stem}) {
 	    $freq->{$stem} = 0 if not defined $freq->{$stem};
 	    $prob{$stem} = (1+$freq->{$stem})/$freq->{_ALL_};
 	    warn "$stem\t$prob{$stem}\n" if $opt{v};
 	}
	for my $ft (split(/[\^\+\s]/, $mtag)) {
	    next if $ft eq '';
	    $ft = '*UNKNOWN*' if $ft eq '***UNKNOWN' or $ft eq 'UNK';
	    next if defined $prob{$ft};
	    die "Model not defined for $ft in $tag" if not defined $model->{$ft};
	    die "Model empty for $ft in $tag" if @{$model->{$ft}} == 0;
	    die "Accuracy not defined for $ft in $tag" if not defined $acc->{$ft};
	    my $guess = predict($model->{$ft}, \%attr);
	    my $p = $acc->{$ft} / 100.0;
	    $p = 0.99 if $p > 0.99;
	    $p = 0.01 if $p < 0.01;
	    $p = 1 - $p if $guess == 0;
	    $prob{$ft} = $p;
	    warn "$ft\t$guess\t$prob{$ft}\n" if $opt{v};
	}
    }
    
#    print STDERR Dumper(\%prob);

    # Evaluate each candidate tag and determine the one with highest probability:
    my $besttag;
    my $bestp;
    for my $tag (@tags) {
	my $p = 0;
	for my $ft (keys %prob) {
	    if ($tag =~ /\b$ft\b/) {
		$p += log($prob{$ft});
	    } elsif (not $opt{p}) {
	        $p += log(1 - $prob{$ft});
	    }
	}
	warn substr($p,0,6) . "\t$tag\n" if $opt{v};

	# TIE-BREAK: If we use 
        # $p > $bestp, the program will select the correct answer
	# $p >= $bestp, the program will select the wrong answer
	# in case of a tie

	if (not defined $bestp or $p > $bestp) {
	    $besttag = $tag;
	    $bestp = $p;
	}
    }
    if ($besttag eq $answer) {
	$stats->{correct}++;
	print "$word $answer\n";
	warn "Found correct: $word $answer\n" if $opt{v};
    } else {
	$stats->{incorrect}++;
	print "$word $besttag\n";
	warn "Found incorrect: $word $answer $besttag\n" if $opt{v};
    }
}

sub predict {
    my ($model, $attr) = @_;
    for (my $i = $#{$model}; $i >= 0; $i--) {
	my @rule = split(/\s+/, $model->[$i]);
	my $guess = shift(@rule);
	my $match = 1;
	for my $a (@rule) {
	    if (not defined $attr->{$a}) {
		$match = 0;
		last;
	    }
	}
	return $guess if $match;
    }
    die "Should never come here\n" . Dumper($model, $attr);
}

print STDERR "\n".Dumper($stats);

