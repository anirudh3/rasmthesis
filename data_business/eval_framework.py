
import argparse
import numpy

def compute_eval_metrics(S, G, thresh):
	print G

	S_bin = S > thresh
	
	print S_bin
	ntp = numpy.sum((S_bin == 1) & (G == 1))
	nfn = numpy.sum((S_bin == 0) & (G == 1))
	ntn = numpy.sum((S_bin == 0) & (G == 0))
	nfp = numpy.sum((S_bin == 1) & (G == 0))

	tpr = ntp*1.0/(ntp+nfn)
	tnr = ntn*1.0/(ntn+nfp)

	return tpr, tnr

def compute_cross_entropy(S, G):
	cross_entropy_loss = numpy.sum(G*numpy.log(S) + (1-G)*numpy.log(1-S))

	return cross_entropy_loss


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Compute goodness score')
	parser.add_argument('-s', type=string, help='numpy file of similarity matrix')
	parser.add_argument('-g', type=string, help='numpy file of ground truth matrix')
	parser.add_argument('--thresh', type=float, help='threshold for similarity_matrix')

	args = parser.parse_args()

	similarity_matrix = numpy.load(args.s)
	ground_truth = numpy.load(args.g)


	cross_entropy = compute_cross_entropy(similarity_matrix, ground_truth)
	tpr, tnr = compute_eval_metrics(similarity_matrix, ground_truth, args.thresh)

	print "True Positive Rate: {} \n" \
	"True Negative Rate: {} \n" \
	"False Positive Rate: {} \n" \
	"False Negative Rate: {} \n" \
	"Cross Entropy Loss: {} \n".format(tpr, tnr, 1-tpr, 1-tnr, cross_entropy)

