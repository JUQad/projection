
#include "loadData.h"
#include "kernel.h"
#include "solver.h"
#include <fstream>
#include <cmath>
#include "omp.h"

//#include "shark/LinAlg/BLAS/remora.hpp"
//#include "shark/LinAlg/BLAS/solve.hpp"
//#include "shark/LinAlg/BLAS/vector_expression.hpp"
//using namespace remora;
using namespace std;
using namespace Eigen;

#define parallel

int no_of_projected_vectors = 600;
void fillbudgetSequence(std::vector<INDEX>& sequence, size_t number_of_points) {
    //unsigned int rand_num = std::rand();
    // srand(rand_num);
    
    for (size_t counter = 0; counter < number_of_points; counter++) {
        //size_t counter_srand = rand()% number_of_points ;
        sequence.push_back(counter);
    }
    random_shuffle(sequence.begin(), sequence.end());
    //random_shuffle(sequence.begin(), sequence.end(), myrandom);
}


// enable exactly one of the following:
#define GSS
 //#define GSS_HIGH_PRECISION
// #define LOOKUP_H
//#define LOOKUP_WD

//#define sameLabel



struct dv_struct {
    double dv_value;
    INDEX dv_index;
};

struct by_value {
    bool operator()(dv_struct const &a, dv_struct const &b) {
        return a.dv_value < b.dv_value;
    }
};

struct by_index {
    bool operator()(dv_struct const &a, dv_struct const &b) {
        return a.dv_index < b.dv_index;
    }
};



std::vector<double> computeS_studentDist(vector<SE>& point, char label, sparseData& dataset, Kernel& kernel) {
    double s_studentDist_sum = 0.0;
    size_t number_of_variables = dataset.data.size();
    vector<double>q;
    for (INDEX j = 0; j < number_of_variables; j++) {
        double q_i = kernel.evaluate(point, dataset.data[j]);
        q.push_back(q_i);
        
        s_studentDist_sum += q_i;
        
    }
    
    for (INDEX j = 0; j < q.size(); j++) {
        q[j] = q[j]/s_studentDist_sum;
    }
    
    return q;
    //return pseudo_gradient;
}



double squaredWeightDegradation(double kernelmn, double kernelmz, double kernelnz, double alpha_m, double alpha_n, double alpha_z) {
	return (alpha_m*alpha_m) * 1.0 // kernel(m, m)
		 + (alpha_n*alpha_n) * 1.0 // kernel(n, n)
		 - (alpha_z*alpha_z) * 1.0 // kernel(z, z)
         + 2*alpha_m*alpha_n * kernelmn;
		//- 2*alpha_m*alpha_z * kernelmz
    	//- 2*alpha_n*alpha_z * kernelnz;
}

double bilinearInterpolation(double q11, double q12, double q21, double q22, double x1, double x2, double y1, double y2, double x, double y){
    double x2x1    = x2 - x1;
    double y2y1    = y2 - y1;
    double x2x     = x2 - x;
    double y2y     = y2 - y;
    double yy1     = y - y1;
    double xx1     = x - x1;
    return  1.0 / (x2x1 * y2y1) * (
                                  q11 * x2x * y2y +
                                  q21 * xx1 * y2y +
                                  q12 * x2x * yy1 +
                                  q22 * xx1 * yy1
                                  );
}

// look up a value from the table with bilinear interpolation
double lookUpTable(vector<double> const& m_table, vector<double> const& k_table, vector<double> const& wd_table, double kernel12, double m)
{
	size_t m_gridDim        = m_table.size();
	double m_gridStepSize   = m_table[1]-m_table[0];

	//double bilinInterpolationValue = 0.0;

	double x = m;
	double y = kernel12;

	double m_initial = m_table[0];
	double k_initial = k_table[0];

	size_t index_m_before = floor( (m-m_initial)/m_gridStepSize);
	size_t index_m_after  = index_m_before + 1;
	size_t index_k_before = floor( (kernel12 - k_initial)/m_gridStepSize);
	size_t index_k_after  = index_k_before + 1;

	double x1  = m_table[index_m_before];
	double x2  = m_table[index_m_after];

	double y1  = k_table[index_k_before];
	double y2  = k_table[index_k_after];

	double q11 = wd_table[index_m_before*m_gridDim + index_k_before];
	double q12 = wd_table[index_m_before*m_gridDim + index_k_after];

	double q22 = wd_table[index_m_after*m_gridDim + index_k_after];
	double q21 = wd_table[index_m_after*m_gridDim + index_k_before];

	return bilinearInterpolation(q11, q12, q21, q22, x1, x2, y1, y2, x, y);
}

// objective function for the Golden Section Search
double objective(double kernel12, double m, double x)
{
	double kernel1z = std::pow(kernel12, (1.0 - x) * (1.0 - x));
	double kernel2z = std::pow(kernel12, x * x);
	return m * kernel1z + (1 - m) * kernel2z;
}

double goldenSectionSearch(double kernel12, double m, double a, double b, double epsilon) {
	double gratio = (sqrt(5.0) - 1.0) / 2.0;
	//double b = 1.0;
	//double a = 0.0;
	double p = b - gratio * (b - a);
	double q = a + gratio * (b - a);

	double fp = objective(kernel12, m, p);
	double fq = objective(kernel12, m, q);
	while ((b - a) >=  epsilon)
	{
		if (fp >= fq)
		{
			b = q;
			q = p;
			fq = fp;

			p = b - gratio * (b - a);
			fp = objective(kernel12, m, p);
		}
		else
		{
			a = p;
			p = q;
			fp = fq;

			q = a + gratio * (b - a);
			fq = objective(kernel12, m, q);
		}
	}

	return ((a + b) / 2.0);
}

tuple<INDEX, INDEX> mergeHeuristicWD(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C)
{
	INDEX index_m = END();
	INDEX index_aux = END();
	double current_min_m = std::numeric_limits<double>::infinity();
	double current_min_aux = std::numeric_limits<double>::infinity();

    // Step 1: finding the first SV with the smallest alpha (line 2 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
	for (INDEX dv_index = 0; dv_index < dual_variables.size(); dv_index++)
    {
		double dv_current_value = abs(dual_variables[dv_index]);
		if (dv_current_value == 0) continue;
		if (dv_current_value < current_min_m) {
			current_min_aux = current_min_m;
			index_aux = index_m;
			current_min_m = dv_current_value;
			index_m = dv_index;
		} else if (dv_current_value < current_min_aux) {
			current_min_aux = dv_current_value;
			index_aux = dv_index;
		}
	}
    //--------------------------------------------------------------------------//
	double alpha_m = dual_variables[index_m];
	std::vector<SE> const& x_m = dataset.data[index_m];
	std::vector<SE> const& x_aux = dataset.data[index_aux];
	char label_m = dataset.labels[index_m];
	char label_aux = dataset.labels[index_aux];
	double alpha_aux = dual_variables[index_aux];

    // Step 2: finding the merge partner based on the WD method
	double min_weight_degradation = std::numeric_limits<double>::infinity();
	INDEX index_n = END();
    double m, alpha_candidate;

	for (INDEX i = 0; i < dual_variables.size(); i++)
    {
       if ((i == index_m) || (label_m != dataset.labels[i])) continue; //same label
        //if(i == index_m) // different label
		alpha_candidate = dual_variables[i];
        /*line 5 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
		m = alpha_m / (alpha_m + alpha_candidate);

		std::vector<SE> const& x_candidate = dataset.data[i];
		double kernel12 = kernel.evaluate(x_m, x_candidate);
        /*iterative method golden section search or bilinear interpolation to compute h(slice D in figure 3 and line 7 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
#ifndef LOOKUP_WD
#ifdef GSS_HIGH_PRECISION
        
        double optimal_h = goldenSectionSearch(kernel12, 0, 1, m, 1e-10); //precise GSS
#endif
#ifdef GSS
        double optimal_h;
        if(label_m == dataset.labels[i])
             optimal_h = goldenSectionSearch(kernel12, m, 0, 1, 0.01); //standard GSS
        else if (alpha_m > alpha_candidate)
             optimal_h = goldenSectionSearch(kernel12, m, 1, 6, 0.01); //standard GSS
        else
             optimal_h = goldenSectionSearch(kernel12, m,-5, 0, 0.01); //standard GSS
#endif
#ifdef LOOKUP_H
        double optimal_h = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12, m); //Lookup-h
#endif
        /*computation of z-coefficient(line 9 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
		double kernel1z = std::pow(kernel12, (1.0 - optimal_h) * (1.0 - optimal_h));
		double kernel2z = std::pow(kernel12, optimal_h * optimal_h);
		double z_coefficient = alpha_m * kernel1z + alpha_candidate * kernel2z;

        /*computation of WD(slice F in figure 3 and line 10 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
        double weight_degradation = squaredWeightDegradation(kernel12, kernel1z, kernel2z, alpha_m, alpha_candidate, z_coefficient);
#else
        //--------------------------------------------------------------------------//
        /*bilinear interpolation to compute WD(slice D in figure 3 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
        //if(m <0) m = std::abs(m);
        double lookup_wd = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12,  m);
        
        lookup_wd *= std::pow((alpha_m + dual_variables[i]), 2);
       
        double weight_degradation = lookup_wd;
        //--------------------------------------------------------------------------//
#endif
        //Evaluate the minimmum weight degredation(line 11 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
		if (weight_degradation < min_weight_degradation)
        {
            /*(line 12 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
			min_weight_degradation = weight_degradation;
            /*(line 13 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
			index_n = i;
		}
	}
 
	if (index_n == END()) {
		// No fitting point with the same label was found, choose the second smallest SV (has different label and will find match guaranteed for B >= 3)
		index_m = index_aux;
		label_m = label_aux;
		alpha_m = alpha_aux;
		vector<SE> const& x_m = x_aux;
		for (INDEX i = 0; i < dual_variables.size(); i++)
		{
			if ((i == index_m) || (label_m != dataset.labels[i])) continue; //same label
            //if (i == index_m)  continue; //different label
			alpha_candidate = dual_variables[i];
			m = alpha_m / (alpha_m + alpha_candidate);
			vector<SE> const& x_candidate = dataset.data[i];
			double kernel12 = kernel.evaluate(x_m, x_candidate);
#ifndef LOOKUP_WD
#ifdef GSS_HIGH_PRECISION
			double optimal_h = goldenSectionSearch(kernel12, m, 0.0, 1.0, 1e-10); //precise GSS
#endif
#ifdef GSS
			double optimal_h = goldenSectionSearch(kernel12, m, 0.0, 1.0, 0.01); //standard GSS
#endif
#ifdef LOOKUP_H
			double optimal_h = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12, m); //Lookup-h
#endif
			/*computation of z-coefficient(line 9 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
			double kernel1z = std::pow(kernel12, (1.0 - optimal_h) * (1.0 - optimal_h));
			double kernel2z = std::pow(kernel12, optimal_h * optimal_h);
			double z_coefficient = alpha_m * kernel1z + alpha_candidate * kernel2z;

			/*computation of WD(slice F in figure 3 and line 10 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
			double weight_degradation = squaredWeightDegradation(kernel12, kernel1z, kernel2z, alpha_m, alpha_candidate, z_coefficient);
#else
			//--------------------------------------------------------------------------//
			/*bilinear interpolation to compute WD(slice D in figure 3 and line 7 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/

			double lookup_wd = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12, m);

			lookup_wd *= std::pow((alpha_m + dual_variables[i]), 2);

			double weight_degradation = lookup_wd;
			//--------------------------------------------------------------------------//
#endif
			//Evaluate the minimmum weight degredation(line 11 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
			if (weight_degradation < min_weight_degradation)
			{
				/*(line 12 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
				min_weight_degradation = weight_degradation;
				/*(line 13 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
				index_n = i;
			}
		}
	}
    std::vector<double> q_matrix;
    //tuple<INDEX, INDEX> mergeHeuristicWD(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C)
   // std::vector<double> computeS_studentDist(vector<SE>& point, char label, vector<double>& pseudo_variables, sparseData& pseudo_data, Kernel& kernel)
    /*vector<double>q;
    double s_studentDist_sum = 0.0;
    for (INDEX j = 0; j < dataset.data.size(); j++) {
        double q_i = kernel.evaluate(dataset.data[index_m], dataset.data[j]);
        q.push_back(q_i);
        
        s_studentDist_sum += q_i;
        
    }
    
    for (INDEX j = 0; j < q.size(); j++) {
        q[j] = q[j]/s_studentDist_sum;
    }*/
    
    //q_matrix=computeS_studentDist(dataset.data[0], dataset.labels[0],  dataset, kernel);
    //cout  << min_weight_degradation << ":";
    
	return std::make_tuple(index_m, index_n);
}


tuple<INDEX, INDEX> mergeHeuristicRandom(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C)
{
    // Select 2 SVs (m,n) for Merging
    INDEX index_m;
    INDEX index_n = END();
    
    // Choose two SVs randomly
    size_t number_of_variables = dual_variables.size();
    index_m = rand() % number_of_variables;
    index_n = rand() % number_of_variables;
    
    char label_m = dataset.labels[index_m];
    char label_n = dataset.labels[index_n];
    
    if(!(label_m == label_n)) {
        INDEX index_aux = rand() % number_of_variables;
        char label_aux = dataset.labels[index_aux];
        if (label_m == label_aux) index_n = index_aux;
        else index_m = index_aux;
    }

    return std::make_tuple(index_m, index_n);
}

tuple<INDEX, INDEX> mergeHeuristicMinAlpha(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C)
{
    // Select 2 SVs (m,n) for Merging
    INDEX index_m = END();
    INDEX index_aux = END();
    double current_min_m = std::numeric_limits<double>::infinity();
    double current_min_aux = std::numeric_limits<double>::infinity();
    
    // Step 1: finding the first SV with the smallest alpha (line 2 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
    for (INDEX dv_index = 0; dv_index < dual_variables.size(); dv_index++)
    {
        double dv_current_value = abs(dual_variables[dv_index]);
        if (dv_current_value == 0) continue;
        if (dv_current_value < current_min_m) {
            current_min_aux = current_min_m;
            index_aux = index_m;
            current_min_m = dv_current_value;
            index_m = dv_index;
        } else if (dv_current_value < current_min_aux) {
            current_min_aux = dv_current_value;
            index_aux = dv_index;
        }
    }
    //--------------------------------------------------------------------------//
    double alpha_m = dual_variables[index_m];
    vector<SE> const& x_m = dataset.data[index_m];
    
    //return std::make_tuple(index_m, index_aux);
    return std::make_tuple(index_m, index_m);
}

tuple<INDEX, INDEX> mergeHeuristicMintwoAlphas(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C)
{
    // Select 2 SVs (m,n) for Merging
    INDEX index_m = END();
    INDEX index_aux = END();
    double current_min_m = std::numeric_limits<double>::infinity();
    double current_min_aux = std::numeric_limits<double>::infinity();
    
    // Step 1: finding the first SV with the smallest alpha (line 2 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
    for (INDEX dv_index = 0; dv_index < dual_variables.size(); dv_index++)
    {
        double dv_current_value = abs(dual_variables[dv_index]);
        if (dv_current_value == 0) continue;
        if (dv_current_value < current_min_m) {
            current_min_aux = current_min_m;
            index_aux = index_m;
            current_min_m = dv_current_value;
            index_m = dv_index;
        } else if (dv_current_value < current_min_aux) {
            current_min_aux = dv_current_value;
            index_aux = dv_index;
        }
    }
    //--------------------------------------------------------------------------//
    unsigned int dv_size = dual_variables.size();
    
    std::vector<dv_struct> dual_variables_abs_struct;
    std::vector<dv_struct> dual_variables_ori_struct;
    dv_struct dv_current_abs_struct;//[dv_size];
    dv_struct dv_current_ori_struct;
    
    std::vector<double> current_wd_vector;
    std::vector<INDEX> current_indexWD_vector;
    double wdValue = INFINITY;
    
    
    /*finding SV with smallest alpha, next SV with smallest alpha, next SV with smallest alpha and different sign*/
    
    for (INDEX dv_index = 0; dv_index < dual_variables.size(); dv_index++)
    {
        double dv_current_value = abs(dual_variables[dv_index]);
        dv_current_abs_struct.dv_value = dv_current_value;
        dv_current_abs_struct.dv_index = dv_index;
        dual_variables_abs_struct.push_back(dv_current_abs_struct);
    }
    
    std::sort(dual_variables_abs_struct.begin(), dual_variables_abs_struct.end(), by_value());
    index_m = dual_variables_abs_struct[0].dv_index;
    INDEX index_oppositeto_m;
    INDEX index_n = END();
    
    
    for (INDEX dv_ind = 1; dv_ind < dual_variables.size(); dv_ind++)
    {
        INDEX index_candidate = dual_variables_abs_struct[dv_ind].dv_index;
        if(dataset.labels[index_m] == dataset.labels[index_candidate])
        {
            index_n = index_candidate;
            break;
        }
    }
    
    if(index_n == END())
    {
        for (INDEX dv_ind = 1; dv_ind < dual_variables.size(); dv_ind++)
        {
            INDEX index_candidate = dual_variables_abs_struct[dv_ind].dv_index;
            if(dataset.labels[index_m] != dataset.labels[index_candidate])
            {
                index_oppositeto_m = index_candidate;
                break;
            }
        }
        
        for (INDEX dv_ind = 1; dv_ind < dual_variables.size(); dv_ind++)
        {
            INDEX index_candidate = dual_variables_abs_struct[dv_ind].dv_index;
            if ((index_oppositeto_m!= index_candidate) && (dataset.labels[index_oppositeto_m] == dataset.labels[index_candidate]))
                {
                    index_m = index_oppositeto_m;
                    index_n = index_candidate;
                    break;
                }
        }
    }
    
    
    
    
    //----------------------------------------------------------------------//
    
    //return std::make_tuple(index_m, index_aux);
    return std::make_tuple(index_m, index_n);
}



tuple<INDEX, INDEX> mergeHeuristicRandomWD(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C)
{
    // Select 2 SVs (m,n) for Merging
    INDEX index_m;
    INDEX index_n;
    INDEX index_aux;
    //----------------------------------------------//
    size_t number_of_variables = dual_variables.size();
    index_m = rand() % number_of_variables;
    index_aux = rand() % number_of_variables;
    //----------------------------------------------//
    double alpha_m = dual_variables[index_m];
    vector<SE> const& x_m = dataset.data[index_m];
    vector<SE> const& x_aux = dataset.data[index_aux];
    char label_m = dataset.labels[index_m];
    char label_aux = dataset.labels[index_aux];
    double alpha_aux = dual_variables[index_aux];
    
    // Step 2: finding the merge partner based on the WD method
    double min_weight_degradation = std::numeric_limits<double>::infinity();
    double m, alpha_candidate;
    
    for (INDEX i = 0; i < dual_variables.size(); i++)
    {
        if ((i == index_m) || (label_m != dataset.labels[i])) continue; //same label
        //if(i == index_m) // different label
        alpha_candidate = dual_variables[i];
        /*line 5 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
        m = alpha_m / (alpha_m + alpha_candidate);
        
        vector<SE> const& x_candidate = dataset.data[i];
        double kernel12 = kernel.evaluate(x_m, x_candidate);
        /*iterative method golden section search or bilinear interpolation to compute h(slice D in figure 3 and line 7 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
#ifndef LOOKUP_WD
#ifdef GSS_HIGH_PRECISION
        
        double optimal_h = goldenSectionSearch(kernel12, 0, 1, m, 1e-10); //precise GSS
#endif
#ifdef GSS
        double optimal_h;
        if(label_m == dataset.labels[i])
            optimal_h = goldenSectionSearch(kernel12, m, 0, 1, 0.01); //standard GSS
        else if (alpha_m > alpha_candidate)
            optimal_h = goldenSectionSearch(kernel12, m, 1, 6, 0.01); //standard GSS
        else
            optimal_h = goldenSectionSearch(kernel12, m,-5, 0, 0.01); //standard GSS
#endif
#ifdef LOOKUP_H
        double optimal_h = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12, m); //Lookup-h
#endif
        /*computation of z-coefficient(line 9 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
        double kernel1z = std::pow(kernel12, (1.0 - optimal_h) * (1.0 - optimal_h));
        double kernel2z = std::pow(kernel12, optimal_h * optimal_h);
        double z_coefficient = alpha_m * kernel1z + alpha_candidate * kernel2z;
        
        /*computation of WD(slice F in figure 3 and line 10 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
        double weight_degradation = squaredWeightDegradation(kernel12, kernel1z, kernel2z, alpha_m, alpha_candidate, z_coefficient);
#else
        //--------------------------------------------------------------------------//
        /*bilinear interpolation to compute WD(slice D in figure 3 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
        
        double lookup_wd = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12,  m);
        
        lookup_wd *= std::pow((alpha_m + dual_variables[i]), 2);
        
        double weight_degradation = lookup_wd;
        //--------------------------------------------------------------------------//
#endif
        //Evaluate the minimmum weight degredation(line 11 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
        if (weight_degradation < min_weight_degradation)
        {
            /*(line 12 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
            min_weight_degradation = weight_degradation;
            /*(line 13 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
            index_n = i;
        }
    }
    
    if (index_n == END()) {
        // No fitting point with the same label was found, choose the second smallest SV (has different label and will find match guaranteed for B >= 3)
        index_m = index_aux;
        label_m = label_aux;
        alpha_m = alpha_aux;
        vector<SE> const& x_m = x_aux;
        for (INDEX i = 0; i < dual_variables.size(); i++)
        {
            if ((i == index_m) || (label_m != dataset.labels[i])) continue; //same label
            //if (i == index_m)  continue; //different label
            alpha_candidate = dual_variables[i];
            m = alpha_m / (alpha_m + alpha_candidate);
            vector<SE> const& x_candidate = dataset.data[i];
            double kernel12 = kernel.evaluate(x_m, x_candidate);
#ifndef LOOKUP_WD
#ifdef GSS_HIGH_PRECISION
            double optimal_h = goldenSectionSearch(kernel12, m, 0.0, 1.0, 1e-10); //precise GSS
#endif
#ifdef GSS
            double optimal_h = goldenSectionSearch(kernel12, m, 0.0, 1.0, 0.01); //standard GSS
#endif
#ifdef LOOKUP_H
            double optimal_h = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12, m); //Lookup-h
#endif
            /*computation of z-coefficient(line 9 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
            double kernel1z = std::pow(kernel12, (1.0 - optimal_h) * (1.0 - optimal_h));
            double kernel2z = std::pow(kernel12, optimal_h * optimal_h);
            double z_coefficient = alpha_m * kernel1z + alpha_candidate * kernel2z;
            
            /*computation of WD(slice F in figure 3 and line 10 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
            double weight_degradation = squaredWeightDegradation(kernel12, kernel1z, kernel2z, alpha_m, alpha_candidate, z_coefficient);
#else
            //--------------------------------------------------------------------------//
            /*bilinear interpolation to compute WD(slice D in figure 3 and line 7 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
            
            double lookup_wd = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12, m);
            
            lookup_wd *= std::pow((alpha_m + dual_variables[i]), 2);
            
            double weight_degradation = lookup_wd;
            //--------------------------------------------------------------------------//
#endif
            //Evaluate the minimmum weight degredation(line 11 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
            if (weight_degradation < min_weight_degradation)
            {
                /*(line 12 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
                min_weight_degradation = weight_degradation;
                /*(line 13 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
                index_n = i;
            }
        }
    }
    std::vector<double> q_matrix;
    //tuple<INDEX, INDEX> mergeHeuristicWD(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C)
    // std::vector<double> computeS_studentDist(vector<SE>& point, char label, vector<double>& pseudo_variables, sparseData& pseudo_data, Kernel& kernel)
    /*vector<double>q;
     double s_studentDist_sum = 0.0;
     for (INDEX j = 0; j < dataset.data.size(); j++) {
     double q_i = kernel.evaluate(dataset.data[index_m], dataset.data[j]);
     q.push_back(q_i);
     
     s_studentDist_sum += q_i;
     
     }
     
     for (INDEX j = 0; j < q.size(); j++) {
     q[j] = q[j]/s_studentDist_sum;
     }*/
    
    //q_matrix=computeS_studentDist(dataset.data[0], dataset.labels[0],  dataset, kernel);
    //cout  << min_weight_degradation << ":";
    
    
    
    
    //--------------------------------------------------------------------------//


    return std::make_tuple(index_m, index_n);
}

tuple<INDEX, INDEX> mergeHeuristic59plusWD(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C)
{
    INDEX index_m = END();
    INDEX index_aux = END();
    double current_min_m = std::numeric_limits<double>::infinity();
    double current_min_aux = std::numeric_limits<double>::infinity();
    
    // Step 1: finding the first SV with the smallest alpha (line 2 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
    for (INDEX dv_index = 0; dv_index < dual_variables.size(); dv_index++)
    {
        double dv_current_value = abs(dual_variables[dv_index]);
        if (dv_current_value == 0) continue;
        if (dv_current_value < current_min_m) {
            current_min_aux = current_min_m;
            index_aux = index_m;
            current_min_m = dv_current_value;
            index_m = dv_index;
        } else if (dv_current_value < current_min_aux) {
            current_min_aux = dv_current_value;
            index_aux = dv_index;
        }
    }
    //--------------------------------------------------------------------------//
    double alpha_m = dual_variables[index_m];
    vector<SE> const& x_m = dataset.data[index_m];
    vector<SE> const& x_aux = dataset.data[index_aux];
    char label_m = dataset.labels[index_m];
    char label_aux = dataset.labels[index_aux];
    double alpha_aux = dual_variables[index_aux];
    
    // Step 2: finding the merge partner based on the WD method
    double min_weight_degradation = std::numeric_limits<double>::infinity();
    INDEX index_n = END();
    double m, alpha_candidate;
    
    //vector<INDEX> sequence(0);
    //sequence.clear();
    //fillbudgetSequence(sequence, dual_variables.size());
    //size_t sequence_size = 59; //sequence.size();
    
    
    for (INDEX i = 0; i < 59; i++)
    {
        //if (i == 59) break;
        //INDEX ws = sequence[i];
        INDEX ws = rand() % dual_variables.size();//dual_variables[i];
        if ((ws == index_m) || (label_m != dataset.labels[ws])) continue; //same label
        //if(i == index_m) // different label
        alpha_candidate = dual_variables[ws];
        /*line 5 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
        m = alpha_m / (alpha_m + alpha_candidate);
        
        vector<SE> const& x_candidate = dataset.data[ws];
        double kernel12 = kernel.evaluate(x_m, x_candidate);
        /*iterative method golden section search or bilinear interpolation to compute h(slice D in figure 3 and line 7 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
#ifndef LOOKUP_WD
#ifdef GSS_HIGH_PRECISION
        
        double optimal_h = goldenSectionSearch(kernel12, 0, 1, m, 1e-10); //precise GSS
#endif
#ifdef GSS
        double optimal_h;
            optimal_h = goldenSectionSearch(kernel12, m, 0, 1, 0.01); //standard GSS
#endif
#ifdef LOOKUP_H
        double optimal_h = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12, m); //Lookup-h
#endif
        /*computation of z-coefficient(line 9 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
        double kernel1z = std::pow(kernel12, (1.0 - optimal_h) * (1.0 - optimal_h));
        double kernel2z = std::pow(kernel12, optimal_h * optimal_h);
        double z_coefficient = alpha_m * kernel1z + alpha_candidate * kernel2z;
        
        /*computation of WD(slice F in figure 3 and line 10 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
        double weight_degradation = squaredWeightDegradation(kernel12, kernel1z, kernel2z, alpha_m, alpha_candidate, z_coefficient);
#else
        //--------------------------------------------------------------------------//
        /*bilinear interpolation to compute WD(slice D in figure 3 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
        
        double lookup_wd = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12,  m);
        
        lookup_wd *= std::pow((alpha_m + dual_variables[i]), 2);
        
        double weight_degradation = lookup_wd;
        //--------------------------------------------------------------------------//
#endif
        //Evaluate the minimmum weight degredation(line 11 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
        if (weight_degradation < min_weight_degradation)
        {
            /*(line 12 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
            min_weight_degradation = weight_degradation;
            /*(line 13 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
            index_n = ws;
        }
    }
    
    if (index_n == END()) {
        // No fitting point with the same label was found, choose the second smallest SV (has different label and will find match guaranteed for B >= 3)
        index_m = index_aux;
        label_m = label_aux;
        alpha_m = alpha_aux;
        vector<SE> const& x_m = x_aux;
        
        //vector<INDEX> sequence(0);
        //sequence.clear();
        //fillbudgetSequence(sequence, dual_variables.size());
        //size_t sequence_size = 59; //sequence.size();
        
        for (INDEX i = 0; i < 59; i++)
        {
            //if (i == 59) break;
            //INDEX ws = sequence[i];
            INDEX ws = rand() % dual_variables.size(); //dual_variables[i];
            if ((ws == index_m) || (label_m != dataset.labels[ws])) continue; //same label
            //if (i == index_m)  continue; //different label
            alpha_candidate = dual_variables[ws];
            m = alpha_m / (alpha_m + alpha_candidate);
            vector<SE> const& x_candidate = dataset.data[ws];
            double kernel12 = kernel.evaluate(x_m, x_candidate);
#ifndef LOOKUP_WD
#ifdef GSS_HIGH_PRECISION
            double optimal_h = goldenSectionSearch(kernel12, m, 0.0, 1.0, 1e-10); //precise GSS
#endif
#ifdef GSS
            double optimal_h = goldenSectionSearch(kernel12, m, 0.0, 1.0, 0.01); //standard GSS
#endif
#ifdef LOOKUP_H
            double optimal_h = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12, m); //Lookup-h
#endif
            /*computation of z-coefficient(line 9 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
            double kernel1z = std::pow(kernel12, (1.0 - optimal_h) * (1.0 - optimal_h));
            double kernel2z = std::pow(kernel12, optimal_h * optimal_h);
            double z_coefficient = alpha_m * kernel1z + alpha_candidate * kernel2z;
            
            /*computation of WD(slice F in figure 3 and line 10 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
            double weight_degradation = squaredWeightDegradation(kernel12, kernel1z, kernel2z, alpha_m, alpha_candidate, z_coefficient);
#else
            //--------------------------------------------------------------------------//
            /*bilinear interpolation to compute WD(slice D in figure 3 and line 7 in algorithm 2 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
            
            double lookup_wd = lookUpTable(get<0>(wd_parameters), get<1>(wd_parameters), get<2>(wd_parameters), kernel12, m);
            
            lookup_wd *= std::pow((alpha_m + dual_variables[i]), 2);
            
            double weight_degradation = lookup_wd;
            //--------------------------------------------------------------------------//
#endif
            //Evaluate the minimmum weight degredation(line 11 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
            if (weight_degradation < min_weight_degradation)
            {
                /*(line 12 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
                min_weight_degradation = weight_degradation;
                /*(line 13 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
                index_n = ws;
            }
        }
    }
    std::vector<double> q_matrix;
    //tuple<INDEX, INDEX> mergeHeuristicWD(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C)
    // std::vector<double> computeS_studentDist(vector<SE>& point, char label, vector<double>& pseudo_variables, sparseData& pseudo_data, Kernel& kernel)
    /*vector<double>q;
     double s_studentDist_sum = 0.0;
     for (INDEX j = 0; j < dataset.data.size(); j++) {
     double q_i = kernel.evaluate(dataset.data[index_m], dataset.data[j]);
     q.push_back(q_i);
     
     s_studentDist_sum += q_i;
     
     }
     
     for (INDEX j = 0; j < q.size(); j++) {
     q[j] = q[j]/s_studentDist_sum;
     }*/
    
    //q_matrix=computeS_studentDist(dataset.data[0], dataset.labels[0],  dataset, kernel);
    //cout  << min_weight_degradation << ":";
    
    return std::make_tuple(index_m, index_n);
}





tuple<INDEX, INDEX> mergeHeuristicKernel(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C)
{
    // Select 2 SVs (m,n) for Merging
    INDEX index_m = END();
    INDEX index_aux = END();
    double current_min_m = std::numeric_limits<double>::infinity();
    double current_min_aux = std::numeric_limits<double>::infinity();
    
    // Search for smallest absolute alpha to merge
    for (INDEX dv_index = 0; dv_index < dual_variables.size(); dv_index++) {
        double dv_current_value = abs(dual_variables[dv_index]);
        if (dv_current_value == 0) continue;
        if (dv_current_value < current_min_m) {
            current_min_aux = current_min_m;
            index_aux = index_m;
            current_min_m = dv_current_value;
            index_m = dv_index;
        } else if (dv_current_value < current_min_aux) {
            current_min_aux = dv_current_value;
            index_aux = dv_index;
        }
    }
    
    vector<SE> x_m = dataset.data[index_m];
    vector<SE> x_aux = dataset.data[index_aux];
    char label_m = dataset.labels[index_m];
    char label_aux = dataset.labels[index_aux];
    
    // Now that the minimum alpha is found, try to find second alpha to merge
    // (Combination with minimum weight degradation)
    
    double max_kernel = -std::numeric_limits<double>::infinity();
    double k_tmp;
    INDEX index_n = END();
    vector<SE> z;
    
    for (INDEX i = 0; i < dual_variables.size(); i++) {
        if ((i == index_m) || (label_m != dataset.labels[i])) continue;
        vector<SE> x_candidate = dataset.data[i];
        k_tmp = kernel.evaluate(x_m, x_candidate);
        if (k_tmp > max_kernel) {
            max_kernel = k_tmp;
            index_n = i;
        }
    }
    if (index_n == END()) {
        // No fitting point with the same label was found, choose the second smallest SV (has different label and will find match guaranteed for B >= 3)
        index_m = index_aux;
        label_m = label_aux;
        x_m = x_aux;
        for (INDEX i = 0; i < dual_variables.size(); i++) {
            if ((i == index_m) || (label_m != dataset.labels[i])) continue;
            vector<SE> x_candidate = dataset.data[i];
            k_tmp = kernel.evaluate(x_m, x_candidate);
            if (k_tmp > max_kernel) {
                max_kernel = k_tmp;
                index_n = i;
            }
        }
    }
    
     return std::make_tuple(index_m, index_n);
}



int mergeAndDeleteSV(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters,double C, Heuristic heuristic) {
	//activate the budget maintenance method and get the indices for merging
	INDEX index_m, index_n;
	std::tie(index_m, index_n) = heuristic(dual_variables, dataset, kernel, wd_parameters, C);

	char label              = dataset.labels[index_m];
	double alpha_m          = dual_variables[index_m];
	double alpha_n          = dual_variables[index_n];
	double m                = alpha_m / (alpha_m + alpha_n);
	vector<SE> const& x_m          = dataset.data[index_m];
	vector<SE> const& x_n          = dataset.data[index_n];
    
	//construct z from the merging partners
	double kernel12 = kernel.evaluate(x_m, x_n);
	double optimal_h        = goldenSectionSearch(kernel12, m, 0.0, 1.0, 0.001);
	vector<SE> z            = scaleAddSparseVectors_new(x_m, x_n, optimal_h, 1-optimal_h);
	double k_mz             = std::pow(kernel12, (1.0 - optimal_h) * (1.0 - optimal_h));
	double k_nz             = std::pow(kernel12, optimal_h * optimal_h);
	double z_coefficient    = alpha_m*k_mz + alpha_n*k_nz;
    
    /*size_t number_of_variables = dual_variables.size();
    double pseudo_gradient = 0.0;
    for (INDEX j = 0; j < number_of_variables; j++) {
        
        pseudo_gradient += dual_variables[j]*kernel12*dataset.labels[j];
    }
    pseudo_gradient *= label;
    pseudo_gradient  = 1 - pseudo_gradient;
    
   z_coefficient =  max(0.0, min(0 + pseudo_gradient, C));*/
   // z_coefficient =  min( m*k_mz + (1-m)*k_nz  , z_coefficient);
   
   if (label != dataset.labels[index_n]) {
        
        //cout << (double)dataset.labels[index_m] << ": " << (double)dataset.labels[index_n] << endl;
    //    double kernel12 = kernel.evaluate(x_m, x_n);
        //cout << kernel12;
       cout << 1;
    }
    
   // cout << "dataset size: " << dataset.labels.size() << " z_coefficient: "  << z_coefficient << " index n: "  << index_n <<  " index m: "  << index_m << " kernel12: " << kernel12 << " z size: " << z.size() << endl;
	//delete old SVs
	dual_variables[index_m] = dual_variables.back();
	dual_variables[index_n] = dual_variables[dual_variables.size() - 2];
	dual_variables.pop_back();
	dual_variables.pop_back();

	dataset.data[index_m] = dataset.data.back();
	dataset.data[index_n] = dataset.data[dataset.data.size() - 2];
	dataset.data.pop_back();
	dataset.data.pop_back();

	dataset.labels[index_m] = dataset.labels.back();
	dataset.labels[index_n] = dataset.labels[dataset.labels.size() - 2];
	dataset.labels.pop_back();
	dataset.labels.pop_back();

	if (z_coefficient != 0) {
		//Add the created SV and its coefficient
		dataset.data.push_back(z);
		dual_variables.push_back(z_coefficient);
		// Add corresponding label
		dataset.labels.push_back(label);
	} else {
		//cout << "merging has resulted in an exact zero coefficient, both points are removed" << endl;
	}
	return (z_coefficient != 0)? 1:0;
}



int DeleteSV(std::vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters,double C, Heuristic heuristic) {
    //activate the budget maintenance method and get the indices for merging
    INDEX index_m, index_n;
    std::tie(index_m, index_n) = heuristic(dual_variables, dataset, kernel, wd_parameters, C);
    
    char label              = dataset.labels[index_m];
    double alpha_m          = dual_variables[index_m];
    double alpha_n          = dual_variables[index_n];
    double m                = alpha_m / (alpha_m + alpha_n);
    vector<SE> const& x_m          = dataset.data[index_m];
    vector<SE> const& x_n          = dataset.data[index_n];
    
    //construct z from the merging partners
    
    //delete old SVs
    dual_variables[index_m] = dual_variables.back();
    
    dual_variables.pop_back();
    
    dataset.data[index_m] = dataset.data.back();
    
    dataset.data.pop_back();
    
    dataset.labels[index_m] = dataset.labels.back();
    
    dataset.labels.pop_back();
    double z_coefficient = 0.0;
    if (z_coefficient != 0) {
        //Add the created SV and its coefficient
       // dataset.data.push_back(z);
       // dual_variables.push_back(z_coefficient);
        // Add corresponding label
       // dataset.labels.push_back(label);
    } else {
       // cout << "merging has resulted in an exact zero coefficient, both points are removed" << endl;
    }
    return (z_coefficient != 0)? 1:0;
}

tuple<INDEX, INDEX, vector<double>> mergeHeuristicWDVector(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C, vector<double> p)
{
    INDEX index_m = END();
    INDEX index_aux = END();
    double current_min_m = std::numeric_limits<double>::infinity();
    double current_min_aux = std::numeric_limits<double>::infinity();
    
    // Choose two SVs randomly
    size_t number_of_variables = dual_variables.size();
    index_m = rand() % (number_of_variables-1);
    
    //--------------------------------------------------------------------------//
    
    char label_m = dataset.labels[index_m];
    INDEX index_n = END();
    double p_sum = 0.0;
    for (INDEX j = 0; j < p.size()-1; j++) {
        p_sum += p[j];
    }
    
    //vector<double> q_vector;
    std::vector<double> q_matrix;
    
    vector<double>q;
    double q_sum = 0.0;
    double gradient = 0.0;
    double s_studentDist_sum = 0.0;
   // cout << dataset.data.size() << endl;
    for (INDEX j = 0; j < dataset.data.size(); j++) {
        if(j == index_m ) continue;
        double q_i = kernel.evaluate(dataset.data[index_m], dataset.data[j]);
       
        gradient += dual_variables[j]* q_i*dataset.labels[j];
        q.push_back(q_i);
        
        s_studentDist_sum += q_i;
        
    }
   // cout << q.size() << endl;
   
    gradient *= dataset.labels[index_m];
    double gr = 1 - gradient;
    
    for (INDEX j = 0; j < q.size(); j++) {
        q[j] = q[j]/s_studentDist_sum;
         q_sum += q[j];
    }
    
     //cout << "p sum: " << p_sum << "; q sum: " << q_sum << endl;
   // cout <<
    q.push_back(gr);
    double KL_MIN = INFINITY;
    double KL_MAX = -INFINITY;
    
    // cout << endl << "***********  ************************************  ***********" << endl;
    double KL_div_sum = 0.0;
    for (INDEX j = 0; j < q.size()-1; j++) {
        if(j == index_m || (label_m != dataset.labels[j])) continue;
        double KL_divergence = p[j] * log(p[j]/q[j]);
        KL_div_sum += -KL_divergence;
        
        if (KL_divergence < KL_MIN){
            //KL_MAX = KL_divergence; index_n = j;
            KL_MIN = KL_divergence; index_n = j;
            //cout << "index = " << index_n << ", KL_min: " << KL_MIN << endl;
        }
    }
    q.push_back(KL_div_sum);
//    cout << KL_div_sum << endl;
    size_t inx_grad= q.size()-1;
    //cout << "p[inx_grad]: " << p[inx_grad] << ", q[inx_grad]: " << q[inx_grad] << endl;
   // double grad_diff = std::abs(p[inx_grad] - q[inx_grad]);
    
    return std::make_tuple(index_m, index_n, q);
}




double  mergeAndDeleteSV_pVector(std::vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C, Heuristic heuristic, std::vector<double> p)
{
 
    //activate the budget maintenance method and get the indices for merging
    INDEX index_m, index_n;
    std::vector<double> q;
    std::tie(index_m, index_n, q) = mergeHeuristicWDVector(dual_variables, dataset, kernel, wd_parameters, C, p);
    
    char label              = dataset.labels[index_m];
    double alpha_m          = dual_variables[index_m];
    double alpha_n          = dual_variables[index_n];
    double m                = alpha_m / (alpha_m + alpha_n);
    vector<SE> const& x_m          = dataset.data[index_m];
    vector<SE> const& x_n          = dataset.data[index_n];
    
    //construct z from the merging partners
    double kernel12 = kernel.evaluate(x_m, x_n);
    double optimal_h        = goldenSectionSearch(kernel12, m, 0.0, 1.0, 0.001);
    vector<SE> z            = scaleAddSparseVectors_new(x_m, x_n, optimal_h, 1-optimal_h);
    double k_mz             = std::pow(kernel12, (1.0 - optimal_h) * (1.0 - optimal_h));
    double k_nz             = std::pow(kernel12, optimal_h * optimal_h);
    double z_coefficient    = alpha_m*k_mz + alpha_n*k_nz;
    if (label != dataset.labels[index_n]) {
        
        //cout << (double)dataset.labels[index_m] << ": " << (double)dataset.labels[index_n] << endl;
        //    double kernel12 = kernel.evaluate(x_m, x_n);
        //cout << kernel12;
        cout << 1;
    }
    //delete old SVs
    dual_variables[index_m] = dual_variables.back();
    dual_variables[index_n] = dual_variables[dual_variables.size() - 2];
    dual_variables.pop_back();
    dual_variables.pop_back();
    
    dataset.data[index_m] = dataset.data.back();
    dataset.data[index_n] = dataset.data[dataset.data.size() - 2];
    dataset.data.pop_back();
    dataset.data.pop_back();
    
    dataset.labels[index_m] = dataset.labels.back();
    dataset.labels[index_n] = dataset.labels[dataset.labels.size() - 2];
    dataset.labels.pop_back();
    dataset.labels.pop_back();
    
    if (z_coefficient != 0) {
        //Add the created SV and its coefficient
        dataset.data.push_back(z);
        dual_variables.push_back(z_coefficient);
        // Add corresponding label
        dataset.labels.push_back(label);
    } else {
        //cout << "merging has resulted in an exact zero coefficient, both points are removed" << endl;
    }
    return (z_coefficient != 0)? q[q.size()-1] : q[q.size()-1];
}

tuple<INDEX, INDEX, double> findMVPair(vector<char>& labels, vector<double>& gradients, vector<double>& dual_variables, vector<double>& lower_constraints_combined, vector<double>& upper_constraints_combined, double accuracy) {
    // Returns a tuple containing the indices of the Most Violating Pair and a boolean containing the result
    // of a check for Keerthi's optimality criterion (max_yg_up <= min_yg_down + epsilon)
    size_t number_of_points = dual_variables.size();
    INDEX max_yg_up_index = END();
    INDEX min_yg_down_index = END();
    INDEX min_yg_down_index2 = END();
    double max_yg_up = -std::numeric_limits<double>::infinity();
    double min_yg_down = std::numeric_limits<double>::infinity();
    
    for (INDEX i = 0; i < number_of_points; i++) {
        double yalpha_i = (double)labels[i] * dual_variables[i];
        double yg_i = (double)labels[i] * gradients[i];
        if (yalpha_i < upper_constraints_combined[i]) {
            if (max_yg_up < yg_i) {
                max_yg_up = yg_i;
                max_yg_up_index = i;
            }
        }
        if (yalpha_i > lower_constraints_combined[i]) {
            if (min_yg_down > yg_i) {
                min_yg_down = yg_i;
                min_yg_down_index = i;
            }
        }
    }
    
   /* min_yg_down = std::numeric_limits<double>::infinity();
    for (INDEX i = 0; i < number_of_points; i++) {
        double yalpha_i = (double)labels[i] * dual_variables[i];
        double yg_i = (double)labels[i] * gradients[i];
        
        if (yalpha_i > lower_constraints_combined[i] && labels[max_yg_up_index] == labels[i]) {
            if (min_yg_down > yg_i ) {
                min_yg_down = yg_i;
                min_yg_down_index2 = i;
            }
        }
    }*/
    
    assert(min_yg_down_index != END() && max_yg_up_index != END()  );
    
    tuple<INDEX, INDEX ,double> result;
    get<0>(result) = max_yg_up_index;
    //if(min_yg_down_index2 != END())
       // get<1>(result) = min_yg_down_index2;
    //else
    get<1>(result) = min_yg_down_index;
    get<2>(result) = (max_yg_up - min_yg_down );
    return result;
}

tuple<INDEX, INDEX> mergeHeuristicReprocessLASVM(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C)
{
    double accuracy = 0.01;
     size_t number_of_model_points = dual_variables.size();
    vector<double> gradients(number_of_model_points, 1.0);
    
   
    vector<double> lower_constraints_combined(number_of_model_points, 0);
    vector<double> upper_constraints_combined(number_of_model_points, 0);
    for (INDEX i = 0; i < number_of_model_points; i++) {
        if (dataset.labels[i] == 1) {
            lower_constraints_combined[i] = 0;
            upper_constraints_combined[i] = C;
        } else if (dataset.labels[i] == -1) {
            lower_constraints_combined[i] = -C;
            upper_constraints_combined[i] = 0;
        }
    }
    //for(unsigned int i=0;i<max_epochs; i++)
        // while (true)
    //{
        tuple<INDEX, INDEX, double> working_set = findMVPair(dataset.labels, gradients, dual_variables, lower_constraints_combined, upper_constraints_combined, accuracy);
        INDEX i, j, ws;
        i = get<0>(working_set);
        j = get<1>(working_set);
        accuracy = get<2>(working_set);
        double y_i;
        double y_j;
        y_i = dataset.labels[i];
        y_j = dataset.labels[j];
        double A_j = lower_constraints_combined[j];
        double B_i = upper_constraints_combined[i];
        double ya_i = y_i*dual_variables[i];
        double ya_j = y_j*dual_variables[j];
        double yg_i = y_i*gradients[i];
        double yg_j = y_j*gradients[j];
   // }
    
    //perform Direction Search
        double newton_min = (yg_i - yg_j)/(kernel.evaluate(dataset.data[i], dataset.data[i]) + kernel.evaluate( dataset.data[j], dataset.data[j]) - 2*kernel.evaluate(dataset.data[i],  dataset.data[j]));
        double lambda = min(B_i - ya_i, min(ya_j - A_j, newton_min));
    // Gradient Update
        for (INDEX index = 0; index < gradients.size(); index++)
        {
            double gradient_change = lambda*dataset.labels[index]*(kernel.evaluate(dataset.data[j], dataset.data[index]) - kernel.evaluate(dataset.data[i], dataset.data[index]));
            gradients[index] += gradient_change;
        }
    unsigned int number_of_support_vectors = dual_variables.size();
    // Dual Variables Update
    double old_alpha_i = dual_variables[i];
    double old_alpha_j = dual_variables[j];
    
    double new_alpha_i = old_alpha_i + lambda*dataset.labels[i];
    // double new_alpha_i =min(0.0, max(old_alpha_i + lambda*dataset.labels[i], C));
    double new_alpha_j = old_alpha_j - lambda*dataset.labels[j];
    //double new_alpha_j =min(0.0, max(old_alpha_j + lambda*dataset.labels[j], C));
    
    if (old_alpha_i == 0 && new_alpha_i != 0) number_of_support_vectors++;
    if (old_alpha_j == 0 && new_alpha_j != 0) number_of_support_vectors++;
    if (old_alpha_i != 0 && new_alpha_i == 0) number_of_support_vectors--;
    if (old_alpha_j != 0 && new_alpha_j == 0) number_of_support_vectors--;
    //cout << "number_of_support_vectors: " << number_of_support_vectors << "out of total: " <<dual_variables.size() << endl;
    /*
     
     // Dual Variables Update
     double old_alpha_i = dual_variables[i];
     double old_alpha_j = dual_variables[j];
     
     double new_alpha_i = old_alpha_i + lambda*dataset.labels[i];
     // double new_alpha_i =min(0.0, max(old_alpha_i + lambda*dataset.labels[i], C));
     double new_alpha_j = old_alpha_j - lambda*dataset.labels[j];
     //double new_alpha_j =min(0.0, max(old_alpha_j + lambda*dataset.labels[j], C));
     
     if (old_alpha_i == 0 && new_alpha_i != 0) number_of_support_vectors++;
     if (old_alpha_j == 0 && new_alpha_j != 0) number_of_support_vectors++;
     if (old_alpha_i != 0 && new_alpha_i == 0) number_of_support_vectors--;
     if (old_alpha_j != 0 && new_alpha_j == 0) number_of_support_vectors--;
     
     */
    INDEX index_m = i;
    INDEX index_n = j;
    return std::make_tuple(index_m, index_n);
    
}

/////////////////


tuple<INDEX, INDEX> projection_smallestalpha(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C)
{
    INDEX index_m           = END();
    INDEX index_n = END();
    INDEX index_aux = END();
        double current_min_m = std::numeric_limits<double>::infinity();
        double current_min_aux = std::numeric_limits<double>::infinity();
        
        // Search for smallest absolute alpha to merge
        for (INDEX dv_index = 0; dv_index < dual_variables.size(); dv_index++) {
            double dv_current_value = abs(dual_variables[dv_index]);
            if (dv_current_value == 0) continue;
            if (dv_current_value < current_min_m) {
                current_min_aux = current_min_m;
                index_aux = index_m;
                current_min_m = dv_current_value;
                index_m = dv_index;
            } else if (dv_current_value < current_min_aux) {
                current_min_aux = dv_current_value;
                index_aux = dv_index;
            }
        }
        
        // Choose two SVs randomly
        size_t number_of_variables = dual_variables.size();
      //index_m = rand() % number_of_variables;
      index_n = rand() % number_of_variables;
      
      char label_m = dataset.labels[index_m];
      char label_n = dataset.labels[index_n];
      
      if(!(label_m == label_n)) {
          INDEX index_aux = rand() % number_of_variables;
          char label_aux = dataset.labels[index_aux];
          if (label_m == label_aux) index_n = index_aux;
          else index_m = index_aux;
      }
    
    return std::make_tuple(index_m, index_n);
}


////////////////


tuple<INDEX, INDEX> projectron(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters, double C)
{
    INDEX index_m           = END();
    INDEX index_n = END();
    INDEX index_aux = END();
        double current_min_m = std::numeric_limits<double>::infinity();
        double current_min_aux = std::numeric_limits<double>::infinity();
        
        // Search for smallest absolute alpha to merge
        for (INDEX dv_index = 0; dv_index < dual_variables.size(); dv_index++) {
            double dv_current_value = abs(dual_variables[dv_index]);
            if (dv_current_value == 0) continue;
            if (dv_current_value < current_min_m) {
                current_min_aux = current_min_m;
                index_aux = index_m;
                current_min_m = dv_current_value;
                index_m = dv_index;
            } else if (dv_current_value < current_min_aux) {
                current_min_aux = dv_current_value;
                index_aux = dv_index;
            }
        }
        
        
        
   
    vector<SE> const& x_m   = dataset.data[index_m];
    char label_m            = dataset.labels[index_m];
    
    // Step 2: finding the projectron partner based on the WD method
    double max_kernel = -std::numeric_limits<double>::infinity();
    double alpha_candidate;
    
    for (INDEX i = 0; i < dual_variables.size(); i++)
    {
        if ((i == index_m) || (label_m != dataset.labels[i])) continue; //same label
        //if(i == index_m) // different label
        alpha_candidate = dual_variables[i];
        vector<SE> const& x_candidate = dataset.data[i];
        double kernel12 = kernel.evaluate(x_m, x_candidate);
        
        if (kernel12 > max_kernel)
        {
            max_kernel = kernel12;
            index_n = 0;
        }
    }
    if(index_n ==  END())
    {
        index_n =0;
        cout << "no index_n value from projectron\n";
        //INDEX index_m = END();
        //INDEX index_aux = END();
        double current_min_m = std::numeric_limits<double>::infinity();
        double current_min_aux = std::numeric_limits<double>::infinity();
        
        // Search for smallest absolute alpha to merge
        for (INDEX dv_index = 0; dv_index < dual_variables.size(); dv_index++) {
            double dv_current_value = abs(dual_variables[dv_index]);
            if (dv_current_value == 0) continue;
            if (dv_current_value < current_min_m) {
                current_min_aux = current_min_m;
                index_aux = index_m;
                current_min_m = dv_current_value;
                index_m = dv_index;
            } else if (dv_current_value < current_min_aux) {
                current_min_aux = dv_current_value;
                index_aux = dv_index;
            }
        }
        
        vector<SE> x_m = dataset.data[index_m];
        vector<SE> x_aux = dataset.data[index_aux];
        char label_m = dataset.labels[index_m];
        char label_aux = dataset.labels[index_aux];
        
        // Now that the minimum alpha is found, try to find second alpha to merge
        // (Combination with minimum weight degradation)
        
        double max_kernel = -std::numeric_limits<double>::infinity();
        double k_tmp;
        //INDEX index_n = END();
        vector<SE> z;
        
        for (INDEX i = 0; i < dual_variables.size(); i++) {
            if ((i == index_m) || (label_m != dataset.labels[i])) continue;
            vector<SE> x_candidate = dataset.data[i];
            k_tmp = kernel.evaluate(x_m, x_candidate);
            if (k_tmp > max_kernel) {
                max_kernel = k_tmp;
                index_n = i;
            }
        }
        if (index_n == END()) {
            // No fitting point with the same label was found, choose the second smallest SV (has different label and will find match guaranteed for B >= 3)
            index_m = index_aux;
            label_m = label_aux;
            x_m = x_aux;
            for (INDEX i = 0; i < dual_variables.size(); i++) {
                if ((i == index_m) || (label_m != dataset.labels[i])) continue;
                vector<SE> x_candidate = dataset.data[i];
                k_tmp = kernel.evaluate(x_m, x_candidate);
                if (k_tmp > max_kernel) {
                    max_kernel = k_tmp;
                    index_n = i;
                }
            }
        }
    }
    
    return std::make_tuple(index_m, index_n);
}



std::vector<double> projectSVLinearEquations(std::vector<double>& dual_variables_notpseudo, sparseData& dataset_notpseudo,vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters,double C, Heuristic heuristic) {

    std::vector<double>time_projections;
  
    double projectoverh1_start_t = (double)clock() / CLOCKS_PER_SEC;
    
    size_t original_pseudo_size = dual_variables.size();
    
    vector<double> partialbudget_coefficient_vector;
    vector<vector<SE>> partialbudget_vector;
    vector<INDEX> ran_no_proj_vect_indices;
    
    
    //\beta(1:B)  + \beta(B+1:N) * K(x(B+1:N) , x(1:B))
    #ifdef parallel
    #pragma omp parallel for
    
    for(int no_proj_vect =0; no_proj_vect < no_of_projected_vectors; no_proj_vect++) //B
    {
        INDEX ran_no_proj_vect =    no_proj_vect;
        //
        vector<SE> const& x_proj_vect           = dataset.data[ran_no_proj_vect];
        double alpha_proj_vect                  = dual_variables[ran_no_proj_vect];
      
        partialbudget_vector.push_back(x_proj_vect);
        partialbudget_coefficient_vector.push_back(alpha_proj_vect);
        ran_no_proj_vect_indices.push_back(ran_no_proj_vect); //get indices to be projected on the rest of the model
    }
   
    #else
    for(int no_proj_vect =0; no_proj_vect < no_of_projected_vectors; no_proj_vect++) //B
    {
        INDEX ran_no_proj_vect =    no_proj_vect;
        //
        vector<SE> const& x_proj_vect           = dataset.data[ran_no_proj_vect];
        double alpha_proj_vect                  = dual_variables[ran_no_proj_vect];
      
        partialbudget_vector.push_back(x_proj_vect);
        partialbudget_coefficient_vector.push_back(alpha_proj_vect);
        ran_no_proj_vect_indices.push_back(ran_no_proj_vect); //get indices to be projected on the rest of the model
    }
     #endif
    size_t B_partial = partialbudget_coefficient_vector.size(); //size of points to be projected
 
    MatrixX<double> Amatrix(B_partial, B_partial); // K(x(1:B), x(1:B))
    MatrixX<double> B1matrix (dual_variables.size()- B_partial , B_partial);// dimension is (N-B). B
    MatrixX<double> B2matrix (1, dual_variables.size()- B_partial ); //vector 1.(N-B)
    
   // double projectB1mat_end_t = (double)clock() / CLOCKS_PER_SEC;
    double projectAmat_start_t = (double)clock() / CLOCKS_PER_SEC;
    
    time_projections.push_back(projectAmat_start_t-projectoverh1_start_t);//overhead1
    
    
   #ifdef parallel
     
   #pragma omp parallel for
    for (INDEX i = 0; i < partialbudget_coefficient_vector.size(); i++)
        {
             for (INDEX j = 0; j < partialbudget_coefficient_vector.size(); j++)
             {
                 double kernel12 = kernel.evaluate(dataset.data[i] , dataset.data[j]);
                 Amatrix(i,j)=kernel12; //K(x(1:B) , x(1:B))
                 
             }
        }
     
    #else
    for (INDEX i = 0; i < partialbudget_coefficient_vector.size(); i++)
    {
         for (INDEX j = 0; j < partialbudget_coefficient_vector.size(); j++)
         {
             double kernel12 = kernel.evaluate(dataset.data[i] , dataset.data[j]);
             Amatrix(i,j)=kernel12; //K(x(1:B) , x(1:B))
             
         }
    }
    #endif
    
 
    double projectB1mat_start_t = (double)clock() / CLOCKS_PER_SEC;
    
    time_projections.push_back(projectB1mat_start_t - projectAmat_start_t); //A1mat time
    
#ifdef parallel
    #pragma omp parallel for
    for (INDEX i = B_partial; i < dual_variables.size(); i++)  //N - B
    {
        for (INDEX j = 0; j < partialbudget_coefficient_vector.size(); j++)  //B
            
         {
            
             double kernel12 = kernel.evaluate(dataset.data[i] , partialbudget_vector[j]) * dual_variables[i] ;
             B1matrix(i-B_partial,j)=kernel12;
             
         }
    }
#else
    for (INDEX i = B_partial; i < dual_variables.size(); i++)  //N - B
    {
        for (INDEX j = 0; j < partialbudget_coefficient_vector.size(); j++)  //B
            
         {
            
             double kernel12 = kernel.evaluate(dataset.data[i] , partialbudget_vector[j]) * dual_variables[i] ;
             B1matrix(i-B_partial,j)=kernel12;
             
         }
    }
#endif
    
   double projectB2mat_start_t = (double)clock() / CLOCKS_PER_SEC;
    time_projections.push_back(projectB2mat_start_t - projectB1mat_start_t); //B1mat time
    
    
#ifdef parallel
    #pragma omp parallel for
    for (INDEX i = B_partial; i < dual_variables.size(); i++)  //N
    {
        B2matrix(0,i-B_partial) =  dual_variables[i];
    }
#else
   for (INDEX i = B_partial; i < dual_variables.size(); i++)  //N
    {
        B2matrix(0,i-B_partial) =  dual_variables[i];
    }
#endif
    
     
    double projectbeta_start_t = (double)clock() / CLOCKS_PER_SEC;
       time_projections.push_back(projectbeta_start_t - projectB2mat_start_t); //B2mat time
    
    
     
    MatrixX<double> Bmatrix (1, B_partial);
    
    Bmatrix = B1matrix.transpose() * B2matrix.transpose();
   
    MatrixX<double> Betamatrix (1, partialbudget_coefficient_vector.size());
    
    
   
    
#ifdef parallel
    #pragma omp parallel for
    for (INDEX i = 0; i < partialbudget_coefficient_vector.size(); i++)  //B
    {
        Betamatrix(0,i) =  partialbudget_coefficient_vector[i];
    }
#else
    for (INDEX i = 0; i < partialbudget_coefficient_vector.size(); i++)  //B
    {
        Betamatrix(0,i) =  partialbudget_coefficient_vector[i];
    }
#endif
    
    
    double projectoverh2_start_t = (double)clock() / CLOCKS_PER_SEC;
          time_projections.push_back(projectoverh2_start_t - projectbeta_start_t ); //betamat time
    
    
    
    
    
    Bmatrix += (Amatrix * Betamatrix.transpose()) ;  //\beta(1:B) + \beta(B+1:N) * K(B+1:N, 1:B)
   
    MatrixX<double> gamma_matrix ;//(1, B_partial);
    
    gamma_matrix = Amatrix.colPivHouseholderQr().solve(Bmatrix) ; //solve A \gamma = B
    
    
    double projectoverh2_end_t = (double)clock() / CLOCKS_PER_SEC;
    time_projections.push_back(projectoverh2_end_t - projectoverh2_start_t ); //overhead2 time
    
    
   // cout << gamma_matrix.size();
    //the problem here those gamma coefficients are very big numbers.
    
    
    //in this part i have the new gamma coefficients i will only update the model to have these values only and remove others?
    dual_variables.clear();
   // double projectB1mat_start_t = (double)clock() / CLOCKS_PER_SEC;
    
#ifdef parallel
   #pragma omp parallel for
    for (int i = 0; i < B_partial; i++){
       // cout << gamma_matrix(i,0) << endl;
        dual_variables.push_back(gamma_matrix(i,0));
    }
#else
    for (int i = 0; i < B_partial; i++){
       // cout << gamma_matrix(i,0) << endl;
        dual_variables.push_back(gamma_matrix(i,0));
    }
#endif
    
#ifdef parallel
    #pragma omp parallel for
    for (int i = B_partial; i < original_pseudo_size; i++){
      
    dataset.data.pop_back();
    dataset.labels.pop_back();
    }
#else
    for (int i = B_partial; i < original_pseudo_size; i++){
      
    dataset.data.pop_back();
    dataset.labels.pop_back();
    }
#endif
   
  // double projectoverh2_end_t = (double)clock() / CLOCKS_PER_SEC;
    //        time_projections.push_back(projectoverh2_end_t - projectoverh2_start_t ); //overhead2 time
    
   // return (projectB1mat_end_t-projectB1mat_start_t);
    return time_projections;
}

/*

 int projectSVLinearEquations(std::vector<double>& dual_variables_notpseudo, sparseData& dataset_notpseudo,vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters,double C, Heuristic heuristic) {

     INDEX index_m, index_n;
     //std::tie(index_m, index_n) = projectron(dual_variables, dataset, kernel, wd_parameters, C);
     
     std::tie(index_m, index_n) = projection_smallestalpha(dual_variables, dataset, kernel, wd_parameters, C);
     size_t original_pseudo_size = dual_variables.size();
     //today:char label              = dataset.labels[index_m];
     double alpha_m          = dual_variables[index_m];
     //today: double alpha_n          = dual_variables[index_n];
     vector<SE> const& x_m          = dataset.data[index_m];
      //today:  vector<SE> const& x_n          = dataset.data[index_n];
     
     
     
     vector<double> partialbudget_coefficient_vector;
     vector<vector<SE>> partialbudget_vector;
     vector<INDEX> ran_no_proj_vect_indices;
     
     for(int no_proj_vect =0; no_proj_vect < no_of_projected_vectors; no_proj_vect++)
     {
         //size_t random_proj_vect =   dual_variables.size();
         //INDEX ran_no_proj_vect =    rand() % random_proj_vect;
         INDEX ran_no_proj_vect =    no_proj_vect;
         //
         vector<SE> const& x_proj_vect           = dataset.data[ran_no_proj_vect];
         double alpha_proj_vect                  = dual_variables[ran_no_proj_vect];
         //double kernel12                         = kernel.evaluate(x_m, x_proj_vect);
       
         partialbudget_vector.push_back(x_proj_vect);
         partialbudget_coefficient_vector.push_back(alpha_proj_vect);
         ran_no_proj_vect_indices.push_back(ran_no_proj_vect);
     }
     
     
     
     
     
     //delete old SVs
     dual_variables[index_m] = dual_variables.back();
     dual_variables.pop_back();
     
     dataset.data[index_m] = dataset.data.back();
     dataset.data.pop_back();
     
     dataset.labels[index_m] = dataset.labels.back();
     dataset.labels.pop_back();
     
     size_t B_partial = partialbudget_coefficient_vector.size();
     

     //now : remora::matrix<double> Amatrix(B_partial, B_partial);
     //now : remora::matrix<double> B1matrix (dual_variables.size()- B_partial , B_partial);
     //now : remora::matrix<double> B2matrix (1, dual_variables.size()- B_partial );
     
     MatrixX<double> Amatrix(B_partial, B_partial);
     MatrixX<double> B1matrix (dual_variables.size()- B_partial , B_partial);
     MatrixX<double> B2matrix (1, dual_variables.size()- B_partial );
     
     for (INDEX i = 0; i < partialbudget_coefficient_vector.size(); i++)
         {
              for (INDEX j = 0; j < partialbudget_coefficient_vector.size(); j++)
              {
                  double kernel12 = kernel.evaluate(dataset.data[i] , dataset.data[j]);
                  Amatrix(i,j)=kernel12;
                  
              }
         }
     
     
     for (INDEX i = B_partial; i < dual_variables.size(); i++)  //N
     {
         for (INDEX j = 0; j < partialbudget_coefficient_vector.size(); j++)  //B
             
          {
             // if (i == dual_variables_notpseudo.size()-1)break;
              //if (j == dual_variables.size()-1)break;
              double kernel12 = kernel.evaluate(dataset.data[i] , partialbudget_vector[j]) * dual_variables[i] ;
              B1matrix(i-B_partial,j)=kernel12;
            //  i_matrix++;
              
          }
     }
     for (INDEX i = B_partial; i < dual_variables.size(); i++)  //N
     {
         B2matrix(0,i-B_partial) =  dual_variables[i];
     }
     
     
     //now: remora::matrix<double> Bmatrix (1, B_partial);
     MatrixX<double> Bmatrix (1, B_partial);
     
     Bmatrix = B1matrix.transpose() * B2matrix.transpose();  //remora::trans(B1matrix) % remora::trans(B2matrix) ;
    
    //now::  remora::matrix<double> Betamatrix (1, partialbudget_coefficient_vector.size());
     MatrixX<double> Betamatrix (1, partialbudget_coefficient_vector.size());
     for (INDEX i = 0; i < partialbudget_coefficient_vector.size(); i++)  //B
     {
         Betamatrix(0,i) =  partialbudget_coefficient_vector[i];
     }
     
      //now :: Bmatrix += remora::trans(Betamatrix);
     Bmatrix += Betamatrix.transpose();
     
     //for (int i = 0; i < B_partial; i++){
       //  cout << Bmatrix(i,0) << endl;
     //}
     //now :: remora::vector<double> Bvector;//(partialbudget_vector.size(), 1.0);
     
     
     
    //now ::
     // for(int i = 0; i < partialbudget_vector.size(); i++){
       //  Bvector.push_back(Bmatrix(i,0));
     //}
     
     
     
     
     
     //remora::matrix<double> Cc(100, 50);
     // skip: fill C
     // compute a symmetric pos semi-definite matrix A
      // solves Ax=b
    // vector<double> Ainv = remora::inv(A,remora::symm_semi_pos_def());
    // vector<double> c =  solution_ % b;   // solves Ax=b
     
     //inv(A,remora::symm_semi_pos_def()) % b;
     
     
     
     //now :: remora::matrix<double> gamma_matrix (1, B_partial);
     MatrixX<double> gamma_matrix ;//(1, B_partial);
     
     gamma_matrix = Amatrix.colPivHouseholderQr().solve(Bmatrix);
     
     dual_variables.clear();
     for (int i = 0; i < B_partial; i++){
        // cout << gamma_matrix(i,0) << endl;
         dual_variables.push_back(gamma_matrix(i,0));
     }
     for (int i = B_partial; i < original_pseudo_size; i++){
       
     dataset.data.pop_back();
     dataset.labels.pop_back();
     }
    // Amatrix % gamma_matrix = Bmatrix;
     //remora::sum(Bmatrix);
     //gamma_matrix = solve(Amatrix, Bvector, symm_pos_def(),left);
     //gamma_matrix = remora::solve(Amatrix, Bvector, remora::symm_semi_pos_def(), left);
     //gamma_matrix = remora::solve(Amatrix, Bvector, remora::symm_pos_def(), left);//solve(Amatrix, Bvector, remora::symm_pos_def(), left);
     //gamma_matrix = remora::solve(Amatrix, Bmatrix);
     //remora::matrix_matrix_solve<Amatrix, Bmatrix, remora::symm_pos_def, left>; //solve( Amatrix, Bmatrix, left);
     
     //solve(Amatrix, Bmatrix);
   
     //Bmatrix = B2matrix * B1matrix;
     // remora::matrix<double> Cc(100, 50);
     
     // skip: fill C
     // compute a symmetric pos semi-definite matrix A
     //remora::matrix<double> A = Cc % remora::trans(Cc);
     //vector<double> b(100, 1.0);         // all ones vector

     //vector<double> solution = inv(A,symm_semi_pos_def()) % b;
     
     
     
    //today dual_variables[index_n] += z_coefficient;
     
     //return (z_coefficient != 0)? 1:0;
     return (0);
 }
 */





int projectSV(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters,double C, Heuristic heuristic) {

    INDEX index_m, index_n;
    //std::tie(index_m, index_n) = projectron(dual_variables, dataset, kernel, wd_parameters, C);
    
    std::tie(index_m, index_n) = projection_smallestalpha(dual_variables, dataset, kernel, wd_parameters, C);
    //today:char label              = dataset.labels[index_m];
    double alpha_m          = dual_variables[index_m];
    //today: double alpha_n          = dual_variables[index_n];
    vector<SE> const& x_m          = dataset.data[index_m];
     //today:  vector<SE> const& x_n          = dataset.data[index_n];
    
        for (INDEX i = 0; i < dual_variables.size(); i++)
        {
            //if (dataset.labels[index_m] != dataset.labels[i]) continue;
            if(i == index_m)continue;
            double kernel12 = kernel.evaluate(x_m, dataset.data[i]);
            //dual_variables[i]= alpha_m*kernel12;
            double optimal_h        = goldenSectionSearch(kernel12, alpha_m /(alpha_m+dual_variables[i]), 0.0, 1.0, 0.01);
            vector<SE> z            = scaleAddSparseVectors_new( x_m, dataset.data[i],
                                       // alpha_m, alpha_m*kernel12 );
                                                                optimal_h, 1-optimal_h);
            
            double z_coefficient = alpha_m*kernel12;
            //dataset.data.push_back(z);
            //dual_variables.push_back(z_coefficient);
                   // Add corresponding label
            dataset.data.push_back(dataset.data[i]);
            dual_variables.push_back(dual_variables[i]);
            dataset.labels.push_back(dataset.labels[i]);
            
            
            dual_variables[i] = dual_variables.back();
            dual_variables.pop_back();
            
            dataset.data[i] = dataset.data.back();
            dataset.data.pop_back();
            
            dataset.labels[i] = dataset.labels.back();
            dataset.labels.pop_back();
        }
   //today double kernel12 = kernel.evaluate(x_m, dataset.data[0]);
   //today double z_coefficient    = (alpha_m)*kernel12;

   // if (label != dataset.labels[index_n]) {
        
     //   cout << "different labels\n";
    //}
    //delete old SVs
    dual_variables[index_m] = dual_variables.back();
    dual_variables.pop_back();
    
    dataset.data[index_m] = dataset.data.back();
    dataset.data.pop_back();
    
    dataset.labels[index_m] = dataset.labels.back();
    dataset.labels.pop_back();
    
   //today dual_variables[index_n] += z_coefficient;
    
    //return (z_coefficient != 0)? 1:0;
    return (0);
}


int projectAndDeleteSV(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters,double C, Heuristic heuristic) {
    //activate the budget maintenance method and get the indices for merging
    INDEX index_m, index_n;
    std::tie(index_m, index_n) = projection_smallestalpha(dual_variables, dataset, kernel, wd_parameters, C);
    //std::tie(index_m, index_n) = heuristic(dual_variables, dataset, kernel, wd_parameters, C);

    char label              = dataset.labels[index_m];
    double alpha_m          = dual_variables[index_m];
    double alpha_n          = dual_variables[index_n];
    //double m                = alpha_m / (alpha_m + alpha_n);
    vector<SE> const& x_m          = dataset.data[index_m];
    // -vector<SE> const& x_n          = dataset.data[index_n];
    
    
    vector<double> z_coefficient_vector;
    vector<vector<SE>> z_vector;
    vector<INDEX> ran_no_proj_vect_indices;
    
    for(int no_proj_vect =0; no_proj_vect < no_of_projected_vectors; no_proj_vect++)
    {
        size_t random_proj_vect = dual_variables.size();
        INDEX ran_no_proj_vect = rand() % random_proj_vect;
        
        //
        if(dataset.labels[index_m] != dataset.labels[ran_no_proj_vect])
           {
               INDEX ran_no_proj_vect_new = rand() % random_proj_vect;
               if(dataset.labels[index_m] != dataset.labels[ran_no_proj_vect_new])
               {
                   index_m          = ran_no_proj_vect_new;
               }
               else
                   ran_no_proj_vect = ran_no_proj_vect_new;
           }
        label              = dataset.labels[index_m];
        
        //
        vector<SE> const& x_proj_vect           = dataset.data[ran_no_proj_vect];
        double alpha_proj_vect                  = dual_variables[ran_no_proj_vect];
        double kernel12                         = kernel.evaluate(x_m, x_proj_vect);
        double m                                = alpha_m / (alpha_m + alpha_proj_vect);
        double optimal_h                        = goldenSectionSearch(kernel12, m, 0.0, 1.0, 0.001);
        vector<SE> z                            = scaleAddSparseVectors_new(x_m, x_proj_vect, optimal_h, 1-optimal_h);
        double k_mz             = std::pow(kernel12, (1.0 - optimal_h) * (1.0 - optimal_h));
        double k_nz             = std::pow(kernel12, optimal_h * optimal_h);
        double z_coefficient    = alpha_m * k_mz + alpha_proj_vect * k_nz;
        z_vector.push_back(z);
        z_coefficient_vector.push_back(z_coefficient);
        ran_no_proj_vect_indices.push_back(ran_no_proj_vect);
        if (label != dataset.labels[ran_no_proj_vect])
        { //Quick check if the projected/merged labels are not similar!
        //  cout << 1;
        }
    }
    
    
    
    
    //delete old SVs
    //Step1: delete alphas
     dual_variables[index_m] = dual_variables.back();
      for(int no_proj_vect =0; no_proj_vect < no_of_projected_vectors; no_proj_vect++)
      {
          INDEX n_projected = ran_no_proj_vect_indices[no_proj_vect];
          dual_variables[n_projected] = dual_variables[(dual_variables.size()) - (no_proj_vect+2)];
          
     
     }
    // dual_variables[index_n] = dual_variables[dual_variables.size() - 2];
    // dual_variables.pop_back();
    for(int no_proj_vect =0; no_proj_vect < no_of_projected_vectors +1 ; no_proj_vect++)
    {
        dual_variables.pop_back();  //to remove the projected vector + index_m
        
    }
     
    
    //Step2: delete data
    dataset.data[index_m] = dataset.data.back();
    for(int no_proj_vect =0; no_proj_vect < no_of_projected_vectors; no_proj_vect++)
     {
         INDEX n_projected = ran_no_proj_vect_indices[no_proj_vect];
         dataset.data[n_projected] = dataset.data[(dataset.data.size()) - (no_proj_vect+2)];
    }
    //dataset.data[index_n] = dataset.data[dataset.data.size() - 2];
    //dataset.data.pop_back();
    for(int no_proj_vect =0; no_proj_vect < no_of_projected_vectors +1 ; no_proj_vect++)
    {
        dataset.data.pop_back();
    }
    
    //Step3: delete labels
    dataset.labels[index_m] = dataset.labels.back();
    for(int no_proj_vect =0; no_proj_vect < no_of_projected_vectors; no_proj_vect++)
     {
         INDEX n_projected = ran_no_proj_vect_indices[no_proj_vect];
         dataset.labels[n_projected] = dataset.labels[(dataset.labels.size()) - (no_proj_vect+2)];
    }
    //dataset.labels[index_n] = dataset.labels[dataset.data.size() - 2];
    //dataset.labels.pop_back();
    for(int no_proj_vect =0; no_proj_vect < no_of_projected_vectors +1 ; no_proj_vect++)
    {
        dataset.labels.pop_back();
    }
    
    
    if (z_coefficient_vector[0] != 0) {
        //Add the created SV and its coefficient
      for(int no_proj_vect =no_of_projected_vectors-1; no_proj_vect >0; no_proj_vect--)
       {
                  //now:  INDEX i = index_n;
            dataset.data.push_back(z_vector[no_proj_vect-1]);
            dual_variables.push_back(z_coefficient_vector[no_proj_vect-1]);
            // Add corresponding label
            dataset.labels.push_back(label);
        }
        
        z_vector.clear();
        z_coefficient_vector.clear();
        ran_no_proj_vect_indices.clear();
        
    } else {
        //cout << "merging has resulted in an exact zero coefficient, both points are removed" << endl;
    }
    
    
    
    return 0;
}


int multimergeAndDeleteSV(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters,double C, Heuristic heuristic) {
    //activate the budget maintenance method and get the indices for merging
    INDEX index_m, index_n;
    //std::tie(index_m, index_n) = projection_smallestalpha(dual_variables, dataset, kernel, wd_parameters, C);
      std::tie(index_m, index_n) = heuristic(dual_variables, dataset, kernel, wd_parameters, C);

    char label              = dataset.labels[index_m];
    double alpha_m          = dual_variables[index_m];
    double alpha_n          = dual_variables[index_n];
    double m                = alpha_m / (alpha_m + alpha_n);
    vector<SE> const& x_m          = dataset.data[index_m];
    //vector<SE> const& x_n          = dataset.data[index_n];
    
    
    vector<double> z_coefficient_vector;
    vector<vector<SE>> z_vector;
    vector<INDEX> ran_no_proj_vect_indices;
    
    vector<SE> const& x_proj_vect           = dataset.data[index_n];
    double alpha_proj_vect                  = dual_variables[index_n];
    double kernel12                         = kernel.evaluate(x_m, x_proj_vect);
    //double kernel12                         = kernel.evaluate(x_m, x_n);
    double optimal_h                        = goldenSectionSearch(kernel12, m, 0.0, 1.0, 0.001);
    vector<SE> z                            = scaleAddSparseVectors_new(x_m, x_proj_vect, optimal_h, 1-optimal_h);
    //vector<SE> z                            = scaleAddSparseVectors_new(x_m, x_n, optimal_h, 1-optimal_h);
    double k_mz             = std::pow(kernel12, (1.0 - optimal_h) * (1.0 - optimal_h));
    double k_nz             = std::pow(kernel12, optimal_h * optimal_h);
    double z_coefficient    = alpha_m * k_mz + alpha_proj_vect * k_nz;
    //double z_coefficient    = alpha_m * k_mz + alpha_n * k_nz;
    
    z_vector.push_back(z);
    z_coefficient_vector.push_back(z_coefficient);
    ran_no_proj_vect_indices.push_back(index_n);
    
    
    
    for(int no_proj_vect =0; no_proj_vect < no_of_projected_vectors; no_proj_vect++)
    {
        size_t random_proj_vect = dual_variables.size();
        INDEX ran_no_proj_vect = rand() % random_proj_vect;
        
        //
       /* if(dataset.labels[index_m] != dataset.labels[ran_no_proj_vect])
           {
               INDEX ran_no_proj_vect_new = rand() % random_proj_vect;
               if(dataset.labels[index_m] != dataset.labels[ran_no_proj_vect_new])
               {
                   index_m          = ran_no_proj_vect_new;
               }
               else
                   ran_no_proj_vect = ran_no_proj_vect_new;
           }
        label              = dataset.labels[index_m];*/
        
        //
        vector<SE> const& x_proj_vect           = dataset.data[ran_no_proj_vect];
        double alpha_proj_vect                  = dual_variables[ran_no_proj_vect];
        double kernel12                         = kernel.evaluate(x_m, x_proj_vect);
        double m                                = alpha_m / (alpha_m + alpha_proj_vect);
        double optimal_h                        = goldenSectionSearch(kernel12, m, 0.0, 1.0, 0.001);
        vector<SE> z                            = scaleAddSparseVectors_new(x_m, x_proj_vect, optimal_h, 1-optimal_h);
        double k_mz             = std::pow(kernel12, (1.0 - optimal_h) * (1.0 - optimal_h));
        double k_nz             = std::pow(kernel12, optimal_h * optimal_h);
        double z_coefficient    = alpha_m * k_mz + alpha_proj_vect * k_nz;
        z_vector.push_back(z);
        z_coefficient_vector.push_back(z_coefficient);
        ran_no_proj_vect_indices.push_back(ran_no_proj_vect);
        if (label != dataset.labels[ran_no_proj_vect])
        { //Quick check if the projected/merged labels are not similar!
            //cout << 1;
        }
    }
    
   // cout << "dataset size: " << dataset.labels.size() << " z_coefficient: "  << z_coefficient_vector[0] << " index n: "  << index_n <<  " index m: "  << index_m << " kernel12: " << kernel12 << " z size: " << z.size() << endl;
    
    
    //delete old SVs
    //Step1: delete alphas
     dual_variables[index_m] = dual_variables.back();
      for(int no_proj_vect =0; no_proj_vect < no_of_projected_vectors+1; no_proj_vect++)
      {
          INDEX n_projected = ran_no_proj_vect_indices[no_proj_vect];
          //cout << "index_n: " << index_n << " ran_no_proj_vect_indices[no_proj_vect]: " << n_projected << endl;
          
          dual_variables[n_projected] = dual_variables[(dual_variables.size()) - (no_proj_vect+2)];
          
     
     }
    // dual_variables[index_n] = dual_variables[dual_variables.size() - 2];
    // dual_variables.pop_back();
    for(int no_proj_vect =0; no_proj_vect < no_of_projected_vectors +2 ; no_proj_vect++)
    {
        dual_variables.pop_back();  //to remove the projected vector + index_m
        
    }
     
    
    //Step2: delete data
    dataset.data[index_m] = dataset.data.back();
    for(int no_proj_vect =0; no_proj_vect < no_of_projected_vectors+1; no_proj_vect++)
     {
         INDEX n_projected = ran_no_proj_vect_indices[no_proj_vect];
         dataset.data[n_projected] = dataset.data[(dataset.data.size()) - (no_proj_vect+2)];
    }
    //dataset.data[index_n] = dataset.data[dataset.data.size() - 2];
    //dataset.data.pop_back();
    for(int no_proj_vect =0; no_proj_vect < no_of_projected_vectors +2 ; no_proj_vect++)
    {
        dataset.data.pop_back();
    }
    
    //Step3: delete labels
    dataset.labels[index_m] = dataset.labels.back();
    for(int no_proj_vect =0; no_proj_vect < no_of_projected_vectors+1; no_proj_vect++)
     {
         INDEX n_projected = ran_no_proj_vect_indices[no_proj_vect];
         dataset.labels[n_projected] = dataset.labels[(dataset.labels.size()) - (no_proj_vect+2)];
    }
    //dataset.labels[index_n] = dataset.labels[dataset.data.size() - 2];
    //dataset.labels.pop_back();
    for(int no_proj_vect =0; no_proj_vect < no_of_projected_vectors +2 ; no_proj_vect++)
    {
        dataset.labels.pop_back();
    }
    
    
 //   if (z_coefficient_vector[0] != 0) {
        //Add the created SV and its coefficient
       // cout << "dataset size: " << dataset.labels.size() << endl;
      for(int no_proj_vect =no_of_projected_vectors+1; no_proj_vect >0; no_proj_vect--)
       {
                  //now:  INDEX i = index_n;
            dataset.data.push_back(z_vector[no_proj_vect-1]);
            dual_variables.push_back(z_coefficient_vector[no_proj_vect-1]);
            // Add corresponding label
            dataset.labels.push_back(label);
        }
    
    
    
    
    
    
        
       // z_vector.clear();
       // z_coefficient_vector.clear();
       // ran_no_proj_vect_indices.clear();
        
   // } else {
     //   cout << "merging has resulted in an exact zero coefficient, both points are removed" << endl;
    //}
    
    
    
    return 0;
}


tuple<int,  double , std::vector<SE>, char > mergeDeleteAdd(vector<double>& dual_variables, sparseData& dataset, Kernel const& kernel, LookupTable const& wd_parameters,double C, Heuristic heuristic) {
    //activate the budget maintenance method and get the indices for merging
    
   
    
    
    INDEX index_m, index_n;
    std::tie(index_m, index_n) = heuristic(dual_variables, dataset, kernel, wd_parameters, C);
    
    char label              = dataset.labels[index_m];
    double alpha_m          = dual_variables[index_m];
    double alpha_n          = dual_variables[index_n];
    double m                = alpha_m / (alpha_m + alpha_n);
    vector<SE> const& x_m          = dataset.data[index_m];
    vector<SE> const& x_n          = dataset.data[index_n];
    
    //construct z from the merging partners
    double kernel12 = kernel.evaluate(x_m, x_n);
    double optimal_h        = goldenSectionSearch(kernel12, m, 0.0, 1.0, 0.001);
    vector<SE> z            = scaleAddSparseVectors_new(x_m, x_n, optimal_h, 1-optimal_h);
    double k_mz             = std::pow(kernel12, (1.0 - optimal_h) * (1.0 - optimal_h));
    double k_nz             = std::pow(kernel12, optimal_h * optimal_h);
    double z_coefficient    = alpha_m*k_mz + alpha_n*k_nz;
    
    
    //delete old SVs
    dual_variables[index_m] = dual_variables.back();
    dual_variables[index_n] = dual_variables[dual_variables.size() - 2];
    dual_variables.pop_back();
    dual_variables.pop_back();
    
    dataset.data[index_m] = dataset.data.back();
    dataset.data[index_n] = dataset.data[dataset.data.size() - 2];
    dataset.data.pop_back();
    dataset.data.pop_back();
    
    dataset.labels[index_m] = dataset.labels.back();
    dataset.labels[index_n] = dataset.labels[dataset.labels.size() - 2];
    dataset.labels.pop_back();
    dataset.labels.pop_back();
    
    if (z_coefficient != 0) {
        //Add the created SV and its coefficient
        dataset.data.push_back(z);
        dual_variables.push_back(z_coefficient);
        // Add corresponding label
        dataset.labels.push_back(label);
    } else {
        //cout << "merging has resulted in an exact zero coefficient, both points are removed" << endl;
    }
    
    return std::make_tuple(1, alpha_m, x_m, label);
}
