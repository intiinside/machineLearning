# Load libraries
use strict;
use warnings;
use Data::Dump qw(dump);
use List::Util qw(zip min max sum);
use sml;
use AI::MXNet qw(mx);

## CHAPTER 01
# Min Max
sub dataset_minmax {
    my ($self, $dataset) = @_;
    return mx->nd->stack ($dataset->min(axis=>0), $dataset->max(axis=>0), axis=1);
}
sml->add_to_class('dataset_minmax',\&{'dataset_minmax'});

sub normalize_dataset{
    my ($self, $dataset; $minmax)=@_;
    my $min = $minmax->slice_axis(axis=>1,begin=>0, end=>1)->T;
    my $max = $minmax->slice_axis(axis=>1, begin=>1, end=>2)->T;

    $dataset->slice(([(0, $dataset->shape->[0]-1], [0, $dataset->shape->[1] -1]) .= ($dataset-$min)/($max-$min);
}
sml->add_to_class('nomalize_dataset', \&{'normalize_dataset'});

# calcular el min por cada columna
sub column_means{
    my ($self, $dataset)=@_;
    return mx->nd->mean($dataset, axis=>0);
};
sml->add_to_class('column_means',\&{'column_means'});



