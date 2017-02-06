#!/usr/bin/perl -w
# Read parse data and extract features
use strict;
my $radius = 2;

my %ctype = 
    ( '!' => 'EXCL', '"' => 'QUOT', '#' => 'NUM', "\$" => 'DOLLAR',
      '%' => 'PERCNT', '&' => 'AMP', '\'' => 'APOS', '(' => 'LPAR',
      ')' => 'RPAR', '*' => 'AST', '+' => 'PLUS', ',' => 'COMMA', '-'
      => 'HYPHEN', '.' => 'PERIOD', '/' => 'SOL', ':' => 'COLON', ';'
      => 'SEMI', '<' => 'LT', '=' => 'EQUALS', '>' => 'GT', '?' =>
      'QUEST', '@' => 'COMMAT', '[' => 'LSQB', '\\' => 'BSOL', ']' =>
      'RSQB', '^' => 'CIRC', '_' => 'LOWBAR', '`' => 'GRAVE', '{' =>
      'LCUB', '|' => 'VERBAR', '}' => 'RCUB', '~' => 'TILDE');

my @s;				# sentence
my @p;				# parse
my @a;				# alternative parses
while(<>) {
    print STDERR "." if ($.%100000==0);
    my ($word, @parses) = split;
    die "Bad line [$_]" unless @parses;
    my $parse = $parses[0];
    if ($word =~ /^</) {
	die "Bad tag [$_]" unless @parses == 1;
	if (@s) {
	    process(\@s, \@p, \@a);
	    @s = (); @p = (); @a = ();
	}
	print "$parse W==$word # $parse\n";
    } else {
	push @s, $word;
	push @p, $parse;
	push @a, \@parses;
    }
}

sub process {
    my ($s, $p, $a) = @_;
    for (my $i = 0; $i <= $#s; $i++) {
	print $p->[$i];
	for (my $j = -$radius; $j <= $radius; $j++) {
	    my $d = ($j==0 ? 'W' : ($j < 0 ? 'L'.(-$j) : "R$j"));
	    my $n = $i + $j;
	    if (($n < 0) or ($n > $#s)) {
		print " $d=<S>"; next;
	    }
	    for my $f (features($s->[$n], $n)) {
		print " $d=$f";
	    }
	}
	print join(" ", " \#", @{$a->[$i]});
	print "\n";
    }
}

sub features {
    my ($tok, $nw) = @_;
    # token is the word, nw is its position in the sentence

    my %ans;
    my $lc = $tok;
    $lc =~ s/\d/0/g;
    $lc =~ tr/ÇÐIÝÖÞÜ/çðýiöþü/;
    $lc =~ tr/A-Z/a-z/;

    # Output the token
    $ans{"=".$tok}++;
    $ans{"~".$lc}++;

    # Output all suffixes in phonetic form
    my $ph = phonetic($lc);
    my $n = length($ph);
    for (my $i = 1; $i < $n; $i++) {
	$ans{"+".substr($ph, -$i)}++;
    }

    # How about prefixes and other substrings?

    # character type features
    my %types;
    my ($first, $last);
    for my $c (split(//, $tok)) { 
	$last = ctype($c);
	$first = $last if not defined $first;
	$types{$last}++;
    }
    if (($first eq 'UPPER') and
	((length($tok) == 1) or
	 (defined $types{LOWER} and
	  ($types{LOWER} == length($tok) - 1)))) {
	if ($nw == 0) {
	    $ans{STFIRST}++;
	} else {
	    $ans{UCFIRST}++;
	}
    } elsif (scalar(keys %types) == 1) {
	$ans{$last}++;
    } else {
	$types{$first}--;
	$types{$last}--;
	$ans{"${first}0"}++;
	for my $t (keys %types) {
	    next if $types{$t} == 0;
	    $ans{"${t}1"}++;
	}
	$ans{"${last}2"}++;
    }
    return keys %ans;
}


sub ctype {
    my $c = shift;
    my $t;
    if ($c =~ /[a-zöýðçþü]/) {
	$t = "LOWER";
    } elsif ($c =~ /[A-ZÖÝÐÇÞÜ]/) {
	$t = "UPPER";
    } elsif ($c =~ /[0-9]/) {
	$t = "DIGIT";
    } else {
	$t = $ctype{$c};
    }
    if (not defined $t) {
	$t = "UNKNOWN";
	warn "Unknown char [$c][".ord($c)."]\n";
    }
    return $t;
}

sub phonetic {
    my $w = shift;
    # group the vowel harmony etc. variants
    # represent groups by capital letters
    # assume $w is already lowercase
    $w =~ s/[ae]/A/g;
    $w =~ s/[ýiuü]/I/g;
    $w =~ s/[dt]/D/g;
    $w =~ s/[bp]/B/g;
    $w =~ s/[cç]/C/g;
    $w =~ s/[kgð]/K/g;
    return $w;
}
