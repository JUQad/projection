
#include "loadData.h"
#include "kernel.h"
#include "svm.h"
#include "budgetMaintenance.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <fstream>

/*almost equal header*/
//#include <cmath>
#include <limits>
#include <iomanip>
//#include <iostream>
//#include</Users/saharqaadan/Documents/downloadsavedhere/EigenPro-matlab-master/eigen-eigen-b3f3d4950030/Eigen/SVD>
//using Eigen::MatrixXf;

using namespace std;
template<class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
almost_equal(T x, T y, int ulp)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x-y) <= std::numeric_limits<T>::epsilon() * std::abs(x+y) * ulp
    // unless the result is subnormal
    || std::abs(x-y) < std::numeric_limits<T>::min();
}




//using namespace std;
//template<class T>
//typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type

//#define MERGE
//#define FREEZE
#define PROJECTION
//////////////////
#define adult
//#define codrna
//#define covtype
//#define susy
//////////////////
#define CHANGE_RATE 0.3
#define PREF_MIN 0.05
#define PREF_MAX 20.0
#define INF HUGE_VAL
//////////////////




////////////////////
//Shuffling dataset
int myrandom (int counter) {
    //cout << counter << endl;
    // double rand1 = std::rand()%(counter);
    //rand1 =std::rand()%(counter);
    //rand1 =std::rand()%(counter);
    //rand1 =std::rand()%(counter);
    //rand1 =std::rand()%(counter);
    //rand1 =std::rand()%(counter);
    //return rand1; }
    //return std::rand()%(counter);
    
}

void fillUniformSequence(vector<INDEX>& sequence, size_t number_of_points) {
    //unsigned int rand_num = std::rand();
    // srand(rand_num);
    
    for (size_t counter = 0; counter < number_of_points; counter++) {
        //size_t counter_srand = rand()% number_of_points ;
        sequence.push_back(counter);
    }
    random_shuffle(sequence.begin(), sequence.end());
    //random_shuffle(sequence.begin(), sequence.end(), myrandom);
}

/*std::vector<double> computeStochKEigenPro(double lambda1, MatrixXf& V, std::vector<double>& s, double tau, vector<SE>& point, char label, vector<double>& pseudo_variables, sparseData& pseudo_data, Kernel& kernel) {
    std::vector<double> pseudo_gradient_vec;
    double pseudo_gradient = 0;
    size_t number_of_variables = pseudo_variables.size();
    // cout << "Start Eigen: "<< (V).size() << ": " << s.size() << ": " << eigenpro_eta << endl;
    //cout << V(1,1) << endl;
    for (INDEX j = 0; j < number_of_variables; j++) {
        // if(j == 159)
        //   cout << "now start debugging\n" ;
        for (INDEX i = 0; i < s.size(); i++)
        {
            pseudo_gradient = pseudo_variables[j]*kernel.evaluate(point, pseudo_data.data[j])*pseudo_data.labels[j];
            
            //pseudo_gradient = pseudo_gradient * V(j,i) * (1-tau*lambda1)/s[i] *  V(i,j);
            // cout << V.size() << ": " << s.size() << ": " << endl;
            pseudo_gradient = pseudo_gradient*   V(j,i) *(1-tau*lambda1)/s[i];
            
            
        }
        pseudo_gradient *= label;
        pseudo_gradient  = 1 - pseudo_gradient;
        //cout << "pseudo_gradient: " << j << ": " << pseudo_gradient << endl;
        pseudo_gradient_vec.push_back(pseudo_gradient);
        pseudo_gradient = 0.0;
    }
    // pseudo_gradient *= label;
    // pseudo_gradient  = 1 - pseudo_gradient;
    return pseudo_gradient_vec;
}
*/
/*
std::tuple<std::vector<double>, MatrixXf, double> rsvd ( sparseData& dataset, Kernel& kernel, size_t M, unsigned int topK)
//function[s, V, lambda] = rsvd(X, phi, M, k)
{
    //% Subsample randomized SVD based on
    //%    Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp.
    //%    "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions."
    //%    SIAM review 53.2 (2011): 217-288.
    //%
    // [input]
    //%    X: (n_sample, n_feature): feature matrix.
    //%    phi: feature map.
    //%    M: subsample size.
    //%    k: top eigensystem.
    
    //% [output]
    //%    s: top-k eigenvalue vector of phi(X) in desecding order.
    //%    V: top-k eigenvectors of phi(X).
    //%    lambda: (k+1)-th largest eigenvalue.
    
    size_t number_of_training_points = dataset.data.size();
    size_t bs = 100;//512; //1//%512;
    //PXs = {};
    
    //inx = randsample(RandStream('mt19937ar', 'Seed', 1), n, M);
    vector<INDEX> sequence(0);
    //fillUniformSequence(sequence, number_of_training_points);
    sequence.clear();
    for (INDEX i=0; i<number_of_training_points; i++)
    {
        sequence.push_back(i);
    }
    for (unsigned int i=0; i<number_of_training_points; i++)
    {
        //swap(ws_vector[i], ws_vector[rand() % ws_vector.size()]);
        swap(sequence[i], sequence[rand() % number_of_training_points]);
    }
    size_t sequence_size = sequence.size();
    std::vector<INDEX> inx;
    for(INDEX inx_index = 0; inx_index < M; inx_index++)
    {
        inx.push_back(sequence[inx_index]);
    }
    std::vector<double> kerneval_vec;
    MatrixXf m1(inx.size(), number_of_training_points);
    MatrixXf U,S1,V1;
    for(unsigned int inx_index = 0; inx_index < inx.size(); inx_index++)
    {
        for(unsigned int ind_data = 0; ind_data < number_of_training_points ; ind_data++)
        {
            double valtemp = kernel.evaluate(dataset.data[inx_index], dataset.data[ind_data]);
            kerneval_vec.push_back(valtemp);
            m1(inx_index,ind_data) = valtemp;
        }
    }
    
    
    //   for (INDEX sindex = 1; sindex <M; sindex++)
    //   {
    //        eindex = min(sindex + bs -1 , M);
    //        PXs{length(PXs) + 1} = phi(X(inx(sindex:eindex), :));
    //        end
    //      INDEX eindex = min( sindex + bs -1, M);
    
    //       kerneval_vec.push_back(kernel.evaluate(dataset.data[sindex], dataset.data[eindex]));
    //    }
    
    //PX = vertcat(PXs{:});
    
    //A = PX;
    unsigned int d = number_of_training_points;//dataset.data.size(); //size(A, 2);
    unsigned int p = std::min(2 * (topK+1), d);
    //double R = std::rand(d, p);
    std::vector<double> Y = kerneval_vec;// * rand(); //A * R;
    // double svd_start_time = (double)clock() / CLOCKS_PER_SEC;
    Eigen::JacobiSVD<MatrixXf> svd(m1, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // double svd_end_time = (double)clock() / CLOCKS_PER_SEC;
    //cout << "time of computing SVD: " << svd_end_time - svd_start_time << endl;
    U = svd.matrixU();  //Its left singular vectors are the columns of the thin U matrix
    S1 = svd.singularValues();  //Its singular values
    // cout << "Its singular values are:" << std::endl << svd.singularValues() << endl;
    // cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd.matrixV() << std::endl;
    // cout << S1(0) << " : " << S1(2) << endl;
    V1 = svd.matrixV();  //Its right singular vectors are the columns of the thin V matrix
    vector<double> s;
    //topK = 1;
    for(unsigned int topk_inx = 0; topk_inx < topK ; topk_inx++)
    {
        s.push_back(sqrt( d/ M) * S1(topk_inx));
    }
    //W = orth(Y);
    //B = W' * A; % rank(B) <= 2(k+1)
    //[~, S1, V1] = svd(B, 'econ');
    
    //s = sqrt(n / M) * diag(S1(1:k, 1:k));
    //V = V1(:, 1:k);
    //cout << V1.rows() << endl;
    ////cout << V1.cols() << endl;
    //V = V1(0: V1.rows()-1, 0: topK-1 );
    MatrixXf V(V1.rows(),topK);
    for (int out_inx = 0; out_inx < V1.rows(); out_inx++)
    {
        for(unsigned int v1_inx = 0; v1_inx < topK ; v1_inx++)
        {
            V(out_inx, v1_inx) = V1(out_inx, v1_inx);
        }
    }
    double lambda;
    //lambda = sqrt(n / M) * S1(k+1, k+1);
    lambda = sqrt(d/M) * S1(topK+1);
    

    //function[s, V, lambda]
    return std::make_tuple( s, V, lambda );
}

*/






void downgrade(unsigned long oto, vector<double>& pseudo_variables)
{
    for (int i= 0; i < pseudo_variables.size() ; i++ )
        pseudo_variables[i] *= (1.0 - 1.0 / (long double) oto);
};

double computeMargin(vector<SE>& point, char label, vector<double>& pseudo_variables, sparseData& pseudo_data, Kernel& kernel) {
    double pseudo_gradient = 0;
    
    size_t number_of_variables = pseudo_variables.size();
    for (INDEX j = 0; j < number_of_variables; j++) {
        
        pseudo_gradient += pseudo_variables[j]*kernel.evaluate(point, pseudo_data.data[j])*pseudo_data.labels[j];
    }
    pseudo_gradient *= label;
    pseudo_gradient  = 1 - pseudo_gradient;
   
    return pseudo_gradient;
}


std::tuple< int,  double*, int*, vector<double>> scheduling( size_t l, int slen, double prefsum, double* acc , double* pref, int* indexX)
{
    // define schedule
    slen = 0;
    double q = l / prefsum;
    vector<double> ws_vector;
    //cout<< "scheduling values are" <<endl ;
    for (int i=0; i<l; i++)
    {
        double a = acc[i] + q * pref[i];
        //cout<< "   pref at top is    " << pref[i] << endl;
        //cout<< "a of r "   <<    i     << "   is  "<< a << "      slen for  "<<     i   << "   is  "<< slen  << endl;
        int n = (int)floor(a);
        
        for (int j=0; j<n; j++)
        {
            indexX[slen] = i;
            slen++;
        }
        acc[i] = a - n;
        //cout<< "acc of  "<<    i     << "   is  "<< acc[i] << "      slen for  "<<     i   << "   is  "<< slen<< endl;
    }
    
    for (int s=0; s<slen;s++ )
    {
        ws_vector.push_back(indexX[s]);
        //while(index[s]==index[s-1] && s<slen  )
        //{ s++;}
        
        
    }
    /* for (int s=0; s<slen/2;s++ )
     {
     ws_vector.push_back(indexX[s]);
     //while(index[s]==index[s-1] && s<slen  )
     //{ s++;}
     
     
     }*/
    
    for (unsigned int i=0; i<ws_vector.size(); i++)
    {
        //swap(ws_vector[i], ws_vector[rand() % ws_vector.size()]);
        swap(ws_vector[i], ws_vector[rand() % ws_vector.size()]);
    }
    
    
    std::tuple< int,  double*, int*, vector<double>> results;
    get<0>(results) = slen;
    get<1>(results) = acc;
    //pseudo_gradient_vec.push_back(get<1>(results));
    get<2>(results) = indexX;
    get<3>(results) = ws_vector;
    return results;
    
}




double computeGradient(vector<SE>& point, char label, vector<double>& pseudo_variables, sparseData& pseudo_data, Kernel& kernel) {
    double pseudo_gradient = 0;
    double max_partpseudo = -INFINITY;
    size_t number_of_variables = pseudo_variables.size();
    for (INDEX j = 0; j < number_of_variables; j++) {
       
        pseudo_gradient += pseudo_variables[j]*kernel.evaluate(point, pseudo_data.data[j])*pseudo_data.labels[j];
       
    }
    pseudo_gradient *= label;
    pseudo_gradient  = 1 - pseudo_gradient;
     //cout << "max_partpseudo: " << max_partpseudo << endl;
    
    return pseudo_gradient;
}

std::vector<double> computeGradientPseudoMaxMin( double& alpha_point, vector<SE>& point, char label, vector<double>& pseudo_variables, sparseData& pseudo_data, Kernel& kernel) {
    double pseudo_gradient = 0.0;
    double part_m = 0.0;
    
    double max_partpseudo = -INFINITY;
    double min_partpseudo = INFINITY;
    double max_partm = -INFINITY;
    double min_partm = INFINITY;
    double pseudo_m = 0.0;
    
    INDEX jMax, jMin;
    size_t number_of_variables = pseudo_variables.size();
    for (INDEX j = 0; j < number_of_variables; j++) {
        double part_pseudo = pseudo_variables[j]*kernel.evaluate(point, pseudo_data.data[j])*pseudo_data.labels[j];
        //(alpha_point ==0.0)? part_m= -INFINITY:part_m = pseudo_variables[j]/alpha_point;
        pseudo_gradient += part_pseudo; //pseudo_variables[j]*kernel.evaluate(point, pseudo_data.data[j])*pseudo_data.labels[j];
        /*if(alpha_point !=0.0)
        {
            part_m = pseudo_variables[j]/alpha_point;
            pseudo_m += part_m;
        if(part_m > max_partm)
        {
            max_partm = part_m;
            //jMax = j;
        }
        
        if(part_m < min_partm)
        {
            min_partm = part_m;
            //jMin = j;
        }
        }
        */
        
        if(part_pseudo > max_partpseudo)
        {
            max_partpseudo = part_pseudo;
            //jMax = j;
        }
        
        if(part_pseudo < min_partpseudo)
        {
            min_partpseudo = part_pseudo;
            //jMin = j;
        }
    }
    //pseudo_gradient += 0.001*max_partpseudo; //pseudo_variables[jMax]*kernel.evaluate(point, pseudo_data.data[jMax])*pseudo_data.labels[jMax];
    pseudo_gradient *= label;
    pseudo_gradient  = 1 - pseudo_gradient;
    //cout << "max_partpseudo: " << max_partpseudo << endl;
    vector<double> g;
    g.push_back(pseudo_gradient);
    g.push_back(max_partpseudo);
   // g.push_back(1+pseudo_gradient);
    g.push_back(min_partpseudo);
    //g.push_back((double)jMax);
    //g.push_back((double)jMin);
    g.push_back(1+pseudo_gradient);//(max_partm);
   // g.push_back(min_partm);
   // g.push_back(pseudo_m);
    return g;
    //return pseudo_gradient;
}

std::vector<double> computeS_studentDistribution(vector<SE>& point, char label, vector<double>& pseudo_variables, sparseData& pseudo_data, Kernel& kernel) {
    double pseudo_gradient = 0;
    double s_studentDist_sum = 0.0;
    double max_partpseudo = -INFINITY;
    size_t number_of_variables = pseudo_variables.size();
    vector<double>p;
    for (INDEX j = 0; j < number_of_variables; j++) {
        double p_i = kernel.evaluate(point, pseudo_data.data[j]);
        p.push_back(p_i);
        
        s_studentDist_sum += p_i;
       pseudo_gradient += pseudo_variables[j]*p_i *pseudo_data.labels[j];
    }
    
    for (INDEX j = 0; j < p.size(); j++) {
        p[j] = p[j]/s_studentDist_sum;
    }
    
    pseudo_gradient *= label;
    pseudo_gradient  = 1 - pseudo_gradient;
    p.push_back(pseudo_gradient);
    return p;
    //return pseudo_gradient;
}




tuple<INDEX, INDEX, bool> findMVP(vector<char>& labels, vector<double>& gradients, vector<double>& dual_variables, vector<double>& lower_constraints_combined, vector<double>& upper_constraints_combined, double accuracy) {
    // Returns a tuple containing the indices of the Most Violating Pair and a boolean containing the result
    // of a check for Keerthi's optimality criterion (max_yg_up <= min_yg_down + epsilon)
    size_t number_of_points = dual_variables.size();
    INDEX max_yg_up_index = END();
    INDEX min_yg_down_index = END();
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
    assert(min_yg_down_index != END() && max_yg_up_index != END());
    tuple<INDEX, INDEX , bool> result;
    get<0>(result) = max_yg_up_index;
    get<1>(result) = min_yg_down_index;
    get<2>(result) = (max_yg_up - min_yg_down <= accuracy);
    return result;
}


tuple<INDEX, INDEX, bool> findMVPs(vector<char>& labels, vector<double>& gradients, vector<double>& dual_variables, vector<double>& lower_constraints_combined, vector<double>& upper_constraints_combined, double accuracy) {
    // Returns a tuple containing the indices of the Most Violating Pair and a boolean containing the result
    // of a check for Keerthi's optimality criterion (max_yg_up <= min_yg_down + epsilon)
    size_t number_of_points = dual_variables.size();
    INDEX max_yg_up_index = END();
    INDEX min_yg_down_index = END();
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
        /*if (yalpha_i > lower_constraints_combined[i]) {
            if (min_yg_down > yg_i) {
                min_yg_down = yg_i;
                min_yg_down_index = i;
            }
        }*/
        
    }
    min_yg_down_index = rand() % (number_of_points-1);
    if(min_yg_down_index == max_yg_up_index){min_yg_down_index = rand() % (number_of_points-1);}
    
    assert(min_yg_down_index != END() && max_yg_up_index != END());
    tuple<INDEX, INDEX , bool> result;
    get<0>(result) = max_yg_up_index;
    get<1>(result) = min_yg_down_index;
    get<2>(result) = (max_yg_up - min_yg_down <= accuracy);
    return result;
}

tuple<double,double,double> primalObjectiveFunction (double C, vector<double>& pseudo_variables, vector<double>& dual_variables, sparseData& pseudo_data, sparseData& data, Kernel& kernel){
    double primaltemp_minW  = 0.0;
    double primaltemp_Hloss  = 0.0;
    for(unsigned int iIter = 0; iIter < pseudo_variables.size(); iIter++)
    {
        for(unsigned int jIter = 0; jIter < pseudo_variables.size(); jIter++)
        {
            double ai = pseudo_variables[iIter];
            double aj = pseudo_variables[jIter];
            double k = kernel.evaluate(pseudo_data.data[iIter], pseudo_data.data[jIter]);
            double yi = pseudo_data.labels[iIter];
            double yj = pseudo_data.labels[jIter];
            primaltemp_minW += ai * aj * k * yi * yj;
        }
    }
    
    for(unsigned int iIter = 0; iIter < dual_variables.size(); iIter++)
    {
        double margin = 0.0;
        for(unsigned int jIter = 0; jIter < pseudo_variables.size(); jIter++)
        {
            
            double aj = pseudo_variables[jIter];
            double k = kernel.evaluate(data.data[iIter], pseudo_data.data[jIter]);
            double yj = pseudo_data.labels[jIter];
            margin += aj * k * yj;
            //margin += pseudo_variables[jIter]*kernel.evaluate(data.data[iIter], pseudo_data.data[jIter])*pseudo_data.labels[jIter];
        }
        margin *= data.labels[iIter];
        double violation = std::max(0.0, 1 - margin);
        primaltemp_Hloss += violation;
    }
    
    
    //return (0.5*primaltemp_minW + C*(primaltemp_Hloss));
    return std::make_tuple(0.5*primaltemp_minW + C*(primaltemp_Hloss), 0.5*primaltemp_minW, C*(primaltemp_Hloss));
}



tuple<double,double,double> dualObjectiveFunction ( double C, vector<double>& pseudo_variables, vector<double>& dual_variables, sparseData& pseudo_data, sparseData& data, Kernel& kernel){
    double dualVartemp = 0.0;
    double dualtemp  = 0.0;
    for(unsigned int iIter = 0; iIter < dual_variables.size(); iIter++)
    {
        dualVartemp += dual_variables[iIter];
    }
    
    for(unsigned int iIter = 0; iIter < dual_variables.size(); iIter++)
    {
        if (dual_variables[iIter] == 0) continue;
        for(unsigned int jIter = 0; jIter < pseudo_variables.size(); jIter++)
        {
            double ai = dual_variables[iIter];
            double aj = pseudo_variables[jIter];
            double k = kernel.evaluate(data.data[iIter], pseudo_data.data[jIter]);
            double yi = data.labels[iIter];
            double yj = pseudo_data.labels[jIter];
            dualtemp += ai * aj * k * yi * yj;
            
            
        }
    }
    
    
    // return (dualVartemp - 0.5* dualtemp);
    return std::make_tuple(dualVartemp - 0.5* dualtemp, dualVartemp, 0.5* dualtemp);
}

//Budgeted primal solver
SVM BSGD(sparseData& dataset, sparseData& testdataset, double C, Kernel& kernel, LookupTable& wd_parameters, double accuracy, size_t B, size_t max_epochs, Heuristic heuristic)
{
    cout << "Optimisation ... begin \n";
    size_t number_of_training_points = dataset.data.size();
    double lambda = 1.0 / ( (double)number_of_training_points * C);
    //double lambda = 1.0 / ( C);
    cout << "Number of training points: " << number_of_training_points << endl;
    
    sparseData pseudo;
    vector<double> pseudo_variables;
    
    vector<INDEX> sequence(0);
    vector<double> dual_variables(number_of_training_points, 0);
    
    unsigned int numIter = 0;
    //unsigned int iterFreq = 100000;
    //double primalObjFunValue = 0.0 , dualObjFunValue = 0.0;
    //double dualVariable = 0.0 , dual_05_minWsquare = 0.0;
   // double primal_05_minWsquare = 0.0 , primal_C_mul_Hloss = 0.0;
    // rowstoppingparameters.txt    pseudostoppingparameters.txt supportvectorscounter.txt
    
    std::string PATH = "trial2/";
    //system("mkdir \"c:/myfolder\"");
    system("mkdir \"trial2\"");
    cout << "Max Epochs: " << max_epochs << "  Current:";
    ofstream dualObjfn_primalfile;
    dualObjfn_primalfile.open ( PATH + "dualobjective.txt");
    //dualObjfn_primalfile << "Writing this to a file.\n";
    ofstream dualObjfn_primalfile_param;
    dualObjfn_primalfile_param.open (PATH + "dualparameters.txt");
    //dualObjfn_primalfile_param << "Writing this to a file.\n";
    
    ofstream primalObjfn_primalfile;
    primalObjfn_primalfile.open (PATH + "primalobjective.txt");
    //primalObjfn_primalfile << "Writing this to a file.\n";
    ofstream primalObjfn_primalfile_param;
    primalObjfn_primalfile_param.open (PATH + "primalparameters.txt");
    //primalObjfn_primalfile_param << "Writing this to a file.\n";
    ofstream primalObjfn_primalfile_per;
    primalObjfn_primalfile_per.open (PATH + "testaccuracy.txt");
    //primalObjfn_primalfile_per << "Writing this to a file.\n";
    ofstream primalObjfn_primalfile_traint;
    primalObjfn_primalfile_traint.open (PATH + "trainingtime.txt");
    //primalObjfn_primalfile_traint << "Writing this to a file.\n";
    //ofstream pObjfn_pf_traint_ver2;
    //pObjfn_pf_traint_ver2.open ("pObjfn_pf_traint_ver2.txt");
    //pObjfn_pf_traint_ver2 << "Writing this to a file.\n";
    
    ofstream primalObjfn_primalfile_pseudoVariables;
    primalObjfn_primalfile_pseudoVariables.open (PATH + "pseudovariables.txt");
    //primalObjfn_primalfile_pseudoVariables << "Writing this to a file.\n";
    ofstream primalObjfn_primalfile_dualVariables;
    primalObjfn_primalfile_dualVariables.open (PATH + "dualvariables.txt");
    //primalObjfn_primalfile_dualVariables << "Writing this to a file.\n";
    //ofstream primalObjfn_primalfile_pseudoData;
    //primalObjfn_primalfile_pseudoData.open (PATH + "ppseudoData_psolver.txt");
    //primalObjfn_primalfile_pseudoData << "Writing this to a file.\n";
    //ofstream primalObjfn_primalfile_pseudoLabels;
    //primalObjfn_primalfile_pseudoLabels.open (PATH + "ppseudoLabels_psolver.txt");
    
    
    //primalObjfn_primalfile_pseudoLabels << "Writing this to a file.\n";
    ofstream pObjfn_pf_merging;
    pObjfn_pf_merging.open (PATH + "merging.txt");
    //pObjfn_pf_merging << "Writing this to a file.\n";
    
    
    //dobjfn_dfile_pseudoLabels << "Writing this to a file.\n";
    /*ofstream primalobjfn_pfile_gradStepdual;
     primalobjfn_pfile_gradStepdual.open (PATH + "pgS_psolverprimal.txt");
     //dobjfn_dfile_pseudoLabels << "Writing this to a file.\n";
     ofstream primalobjfn_pfile_gradSteppseudo;
     primalobjfn_pfile_gradSteppseudo.open (PATH + "pgS_psolverpseudo.txt");
     //dobjfn_dfile_pseudoLabels << "Writing this to a file.\n";
     ofstream primalobjfn_pfile_dualpseudoCounter;
     primalobjfn_pfile_dualpseudoCounter.open (PATH + "pgS_pdualpseudoCounter.txt");
     */
    
    
    
    
    
    //double objfun_start_time = 0.0;
    //double objfun_end_time = 0.0;
    //double objfunction_timer = 0.0;
    double train_start_t = 0.0;
    double  train_end_t = 0.0;
    //double train_start_t_ver2 = (double)clock() / CLOCKS_PER_SEC;
    //train_start_t = (double)clock() / CLOCKS_PER_SEC;
    double sumOfDualvariables = 0.0;
    double sumOfPseudovariables = 0.0;
    
    for (size_t epoch = 0; epoch < max_epochs; epoch++)
    {
        //dataset.shuffle_ds_dualvec(dual_variables);
        
        cout <<  epoch+1 << ":";
        
        
        sequence.clear();
        //fillUniformSequence(sequence, number_of_training_points);
        //size_t sequence_size = sequence.size();
        for (INDEX i=0; i<number_of_training_points; i++)
        {
            sequence.push_back(i);
        }
        for (unsigned int i=0; i<number_of_training_points; i++)
        {
            //swap(ws_vector[i], ws_vector[rand() % ws_vector.size()]);
            swap(sequence[i], sequence[rand() % number_of_training_points]);
        }
        size_t sequence_size = sequence.size();
        double mergeAndDeleteSV_counter = 0.0;
        unsigned int countMerges = 0;
        train_start_t = (double)clock() / CLOCKS_PER_SEC;
        for (INDEX i = 0; i < sequence_size; i++)
        {
            //if(numIter % iterFreq == 0)
            //{
            //    train_start_t = (double)clock() / CLOCKS_PER_SEC;
            
            //}
            INDEX ws = sequence[i];
            numIter++;
            //+++++++++++++ define & implement modelBsgdMap +++++++++++++//
            
            if (numIter == 1)
            {
                pseudo.data.push_back(dataset.data[ws]);
                pseudo.labels.push_back(dataset.labels[ws]);
                pseudo_variables.push_back( 1.0);
                dual_variables[ws] = 1.0;
                continue;
            }
            
            /*calculation the margin(slice A in figure 3 and line 4 in algorithm 1 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
            double marginViolation = computeMargin(dataset.data[ws], dataset.labels[ws], pseudo_variables, pseudo, kernel);
            
            /*calculation the downgrade step(slice B in figure 3 and line 5 in algorithm 1 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)*/
            downgrade(numIter, pseudo_variables);
            //here for testing 01.05.18 --> downgrade(numIter, dual_variables);
            
            
            /*line 6 in algorithm 1 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search*/
            if ( (marginViolation) > 0.0)  // check margin violation
            {
                //add a new SV to the pseudo model(line 7 in algorithm 1 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
                pseudo_variables.push_back( (1.0 / ((double)numIter * lambda)));
                dual_variables[ws] += (1.0 / ((double)numIter * lambda));
                pseudo.data.push_back(dataset.data[ws]);
                pseudo.labels.push_back(dataset.labels[ws]);
            }
            
            
            
            
            //objfun_start_time = (double)clock() / CLOCKS_PER_SEC;
            while (pseudo.data.size() > B)
            {
                //Check the model size compared to the budget(line 9 in algorithm 1 in the paper: Speeding Up Budgeted Stochastic Gradient Descent with Precomputed Golden Section Search)
                double mergeAndDeleteSV_start_time = (double)clock() / CLOCKS_PER_SEC;
                
                mergeAndDeleteSV(pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);
                
                double mergeAndDeleteSV_end_time = (double)clock() / CLOCKS_PER_SEC;
                
                mergeAndDeleteSV_counter += mergeAndDeleteSV_end_time - mergeAndDeleteSV_start_time;
                countMerges++;
            }
            
            
            
            //here for testing 01.05.18 -->
            /*if(numIter % iterFreq == 0)
             
             {
             train_end_t = (double)clock() / CLOCKS_PER_SEC;
             primalObjfn_primalfile_traint << "epoch :"  << epoch +1  << ":trainingtime.:"  << train_end_t - train_start_t << ":" << endl;
             train_end_t = 0.0;
             train_start_t = 0.0;
             
             //std::tie(primalObjFunValue, primal_05_minWsquare, primal_C_mul_Hloss) = primalObjectiveFunction(C, pseudo_variables, dual_variables, pseudo, dataset, kernel );
             
             //primalObjfn_primalfile <<"epoch:"  <<epoch +1 << ":" << primalObjFunValue << ":" << endl;
             //primalObjfn_primalfile_param << "epoch:"  <<epoch +1  << ":C*HLoss:" << primal_C_mul_Hloss << ":(1/2)minWsquare:" << primal_05_minWsquare << ":" << endl ;
             
             // std::tie(dualObjFunValue, dualVariable, dual_05_minWsquare) = dualObjectiveFunction ( C, pseudo_variables, dual_variables, pseudo, dataset, kernel );
             
             // dualObjfn_primalfile <<"epoch:"  << epoch +1 << ":" << dualObjFunValue << ":" << endl;
             
             // dualObjfn_primalfile_param << "epoch:"  <<epoch +1  << ":sumOfalpha:" << dualVariable << ":(1/2)minWsquare:" << dual_05_minWsquare << ":"  << endl;
             }*/
            /*
             /////////////////////////////
             primalObjfn_primalfile_pseudoVariables << "epoch:"  << epoch +1 << ":" ;
             for(int pseudo_index = 0; pseudo_index < pseudo_variables.size(); pseudo_index++ )
             {
             primalObjfn_primalfile_pseudoVariables   << pseudo_variables[pseudo_index] << ":";
             
             }
             primalObjfn_primalfile_pseudoVariables << endl;
             //////////////////////////////
             //save pseudo data
             primalObjfn_primalfile_pseudoData << "epoch:"  << epoch +1 << ":" ;
             for(int pseudo_index = 0; pseudo_index < pseudo.data.size(); pseudo_index++ )
             {
             
             vector<SE> first_vector = pseudo.data[pseudo_index];
             size_t x1_size = first_vector.size();
             
             for (size_t i = 0; i< x1_size-1; i++){
             primalObjfn_primalfile_pseudoData  << first_vector[i].index <<";"<< first_vector[i].value << ":";
             
             }
             primalObjfn_primalfile_pseudoData  << endl;
             
             primalObjfn_primalfile_pseudoData << " : :";
             
             }primalObjfn_primalfile_pseudoData << endl;
             
             for(int pseudo_index = 0; pseudo_index < pseudo_variables.size(); pseudo_index++ )
             {
             primalObjfn_primalfile_pseudoLabels   << (int)pseudo.labels[pseudo_index] << ":";
             
             }
             primalObjfn_primalfile_pseudoLabels << endl;
             
             //save dual variables
             primalObjfn_primalfile_dualVariables << "epoch:"  << epoch +1 << ":" ;
             
             
             for(int dual_index = 0; dual_index < dual_variables.size(); dual_index++ )
             {
             primalObjfn_primalfile_dualVariables   << dual_variables[dual_index] << ":";
             }
             primalObjfn_primalfile_dualVariables << endl;
             
             
             
             }
             */  //here for testing 01.05.18 -->
            //save dual variables
            /*
             primalObjfn_primalfile_dualVariables << "epoch:"  << epoch +1 << ":" ;
             
             
             for(int dual_index = 0; dual_index < dual_variables.size(); dual_index++ )
             {
             primalObjfn_primalfile_dualVariables   << dual_variables[dual_index] << ":";
             }
             primalObjfn_primalfile_dualVariables << endl;
             
             
             if(numIter%iterFreq == 0)
             {
             
             
             ////////// merging time, training time, performance at each epoch:
             pObjfn_pf_merging << "epoch :"  << epoch +1  << ":mergingtime:" << mergeAndDeleteSV_counter << ":mergingsteps:" << countMerges << endl;
             countMerges = 0;
             //SVM svm(pseudo_variables, pseudo, kernel);
             //primalObjfn_primalfile_per << "epoch :"  << epoch +1  << ": per.:"  << svm.evaluateTestset(testdataset)<< ":" << endl;
             
             //primalObjfn_primalfile_traint << "epoch :"  << epoch +1  << ":trainingtime.:"  << train_end_t - train_start_t << ":" << endl;
             //objfunction_timer = 0.0;
             train_start_t = (double)clock() / CLOCKS_PER_SEC;
             
             }
             
             */
            
        } // end sequence
        
        ///////////////////////////////////////////////////////////////////////////////////////// Objective function routine
        train_end_t = (double)clock() / CLOCKS_PER_SEC;
        pObjfn_pf_merging << "epoch :"  << epoch +1  << ":mergingtime:" << mergeAndDeleteSV_counter << ":mergingsteps:" << countMerges << endl;
        countMerges = 0;
        SVM svm(pseudo_variables, pseudo, kernel);
        primalObjfn_primalfile_per << "epoch :"  << epoch +1  << ": per.:"  << svm.evaluateTestset(testdataset)<< ":" << endl;
        
        primalObjfn_primalfile_traint << "epoch :"  << epoch +1  << ":trainingtime.:"  << train_end_t - train_start_t << ":" << endl;
        //objfun_end_time = (double)clock() / CLOCKS_PER_SEC;
        /*
         std::tie(primalObjFunValue, primal_05_minWsquare, primal_C_mul_Hloss) = primalObjectiveFunction(C, pseudo_variables, dual_variables, pseudo, dataset, kernel );
         
         primalObjfn_primalfile <<"epoch:"  <<epoch +1 << ":" << primalObjFunValue << ":" << endl;
         primalObjfn_primalfile_param << "epoch:"  <<epoch +1  << ":C*HLoss:" << primal_C_mul_Hloss << ":(1/2)minWsquare:" << primal_05_minWsquare << ":" << endl ;
         
         std::tie(dualObjFunValue, dualVariable, dual_05_minWsquare) = dualObjectiveFunction ( C, pseudo_variables, dual_variables, pseudo, dataset, kernel );
         
         dualObjfn_primalfile <<"epoch:"  << epoch +1 << ":" << dualObjFunValue << ":" << endl;
         dualObjfn_primalfile_param << "epoch:"  <<epoch +1  << ":sumOfalpha:" << dualVariable << ":(1/2)minWsquare:" << dual_05_minWsquare << ":"  << endl;
         
         primalObjfn_primalfile_pseudoVariables << "epoch:"  << epoch +1 << ":" ;
         for(int pseudo_index = 0; pseudo_index < pseudo_variables.size(); pseudo_index++ )
         {
         primalObjfn_primalfile_pseudoVariables   << pseudo_variables[pseudo_index] << ":";
         
         }
         primalObjfn_primalfile_pseudoVariables << endl;
         
         //save pseudo data
         primalObjfn_primalfile_pseudoData << "epoch:"  << epoch +1 << ":" ;
         for(int pseudo_index = 0; pseudo_index < pseudo.data.size(); pseudo_index++ )
         {
         
         vector<SE> first_vector = pseudo.data[pseudo_index];
         size_t x1_size = first_vector.size();
         
         for (size_t i = 0; i< x1_size-1; i++){
         primalObjfn_primalfile_pseudoData  << first_vector[i].index <<";"<< first_vector[i].value << ":";
         
         }
         primalObjfn_primalfile_pseudoData  << endl;
         
         primalObjfn_primalfile_pseudoData << " : :";
         
         
         }
         primalObjfn_primalfile_pseudoData << endl;
         
         for(int pseudo_index = 0; pseudo_index < pseudo_variables.size(); pseudo_index++ )
         {
         primalObjfn_primalfile_pseudoLabels   << (int)pseudo.labels[pseudo_index] << ":";
         
         }
         primalObjfn_primalfile_pseudoLabels << endl;
         */
        
        if(epoch == 1 || epoch == 49 || epoch == 99 || epoch == 199 || epoch == 299 || epoch == 399 || epoch == 499 )
        {
           /* std::tie(primalObjFunValue, primal_05_minWsquare, primal_C_mul_Hloss) = primalObjectiveFunction(C, pseudo_variables, dual_variables, pseudo, dataset, kernel );
            
            primalObjfn_primalfile <<"epoch:"  <<epoch +1 << ":" << primalObjFunValue << ":" << endl;
            primalObjfn_primalfile_param << "epoch:"  <<epoch +1  << ":C*HLoss:" << primal_C_mul_Hloss << ":(1/2)minWsquare:" << primal_05_minWsquare << ":" << endl ;
            
            std::tie(dualObjFunValue, dualVariable, dual_05_minWsquare) = dualObjectiveFunction ( C, pseudo_variables, dual_variables, pseudo, dataset, kernel );
            
            dualObjfn_primalfile <<"epoch:"  << epoch +1 << ":" << dualObjFunValue << ":" << endl;
            dualObjfn_primalfile_param << "epoch:"  <<epoch +1  << ":sumOfalpha:" << dualVariable << ":(1/2)minWsquare:" << dual_05_minWsquare << ":"  << endl;
            */
            //save dual variables
            primalObjfn_primalfile_dualVariables << "epoch:"  << epoch +1 << "sumofDualVariables/C" << ":" ;
            
            for(int dual_index = 0; dual_index < dual_variables.size(); dual_index++ )
            {
            //primalObjfn_primalfile_dualVariables   << dual_variables[dual_index] << ":";
                sumOfDualvariables += dual_variables[dual_index] ;
            }
            primalObjfn_primalfile_dualVariables << sumOfDualvariables/C << endl;
            sumOfDualvariables = 0.0;
            
            //objfun_end_time = (double)clock() / CLOCKS_PER_SEC;
            //objfunction_timer += objfun_end_time - objfun_start_time ;
            
            primalObjfn_primalfile_pseudoVariables << "epoch:"  << epoch +1 << "sumofpseudoVariables/C" <<":" ;
            for(int pseudo_index = 0; pseudo_index < pseudo_variables.size(); pseudo_index++ )
            {
                //primalObjfn_primalfile_pseudoVariables   << pseudo_variables[pseudo_index] << ":";
                sumOfPseudovariables += pseudo_variables[pseudo_index] ;
                
            }
            primalObjfn_primalfile_pseudoVariables << sumOfPseudovariables/C << endl;
            sumOfPseudovariables = 0.0;
            
            ////////// merging time, training time, performance at each epoch:
            // pObjfn_pf_merging << "epoch :"  << epoch +1  << ":mergingtime:" << mergeAndDeleteSV_counter << ":mergingsteps:" << countMerges << endl;
            // countMerges = 0;
            //SVM svm(pseudo_variables, pseudo, kernel);
            //primalObjfn_primalfile_per << "epoch :"  << epoch +1  << ": per.:"  << svm.evaluateTestset(testdataset)<< ":" << endl;
            
            //primalObjfn_primalfile_traint << "epoch :"  << epoch +1  << ":trainingtime.:"  << train_end_t - train_start_t << ":" << endl;
            //objfunction_timer = 0.0;
        }
        
        
    }
    dualObjfn_primalfile.close();
    primalObjfn_primalfile.close();
    
    dualObjfn_primalfile_param.close();
    primalObjfn_primalfile_param.close();
    
    primalObjfn_primalfile_per.close();
    primalObjfn_primalfile_traint.close();
    pObjfn_pf_merging.close();
    
    primalObjfn_primalfile_pseudoVariables.close();
    //primalObjfn_primalfile_pseudoData.close();
    //primalObjfn_primalfile_pseudoLabels.close();
    primalObjfn_primalfile_dualVariables.close();
    
    return SVM(pseudo_variables, pseudo, kernel);
}
//Budgeted dual solver
SVM BDCA(sparseData& dataset, sparseData& testdataset, double C, Kernel& kernel, LookupTable& wd_parameters, double accuracy, size_t B, size_t max_epochs, Heuristic heuristic)
{
    cout << "Optimisation ... begin \n";
    size_t number_of_training_points = dataset.data.size();
    cout << "Number of training points: " << number_of_training_points << endl;
   // unsigned int iterFreq = 100000;
    sparseData pseudo;
    vector<double> pseudo_variables;
    
    vector<INDEX> sequence(0);
    vector<double> dual_variables(number_of_training_points, 0);
    std::string budget = to_string(B);
    std::string datasetname;
#ifdef adult
    datasetname = "a";
    cout << "dataset: adult\n";
#endif
#ifdef codrna
    datasetname = "c";
     cout << "dataset: codrna\n";
#endif
#ifdef susy
    datasetname = "s";
    cout << "dataset: susy\n";
#endif
#ifdef covtype
    datasetname = "cov";
    cout << "dataset: covtype\n";
#endif
    
    
    // Main optimization loop
    //double dualObjFunValue = 0.0 , primalObjFunValue = 0.0;
    //double dualVariable = 0.0, dual_05_minWsquare = 0.0;
    //double primal_05_minWsquare = 0.0 , primal_C_mul_Hloss = 0.0;
    std::string PATH;
#ifdef FREEZE
    PATH = "freeze2/";
    //system("mkdir \"a_b500_freeze\"");
     system("mkdir \"freeze2\"") ;
    cout << "Freeze mode\n";
#endif
#ifdef PROJECTION
    PATH = "projection800_600/";
    system("mkdir \"projection800_600\"");
    cout << "Projection mode\n";
#endif
#ifdef MERGE
    PATH = "merge2/";
    system("mkdir \"merge2\"");
    cout << "Merging mode\n";
#endif
    
    cout << "Max Epochs: " << max_epochs << "  Current:";
    ofstream dobjfn_dfile;
    dobjfn_dfile.open (PATH + "dualobjective.txt");
    //dobjfn_dfile << "Writing this to a file.\n";
    ofstream dobjfn_dfile_param;
    dobjfn_dfile_param.open (PATH + "dualparameters.txt");
    //dobjfn_dfile_param << "Writing this to a file.\n";
    ofstream dobjfn_dfile_per;
    dobjfn_dfile_per.open (PATH + "testaccuracy.txt");//dualparameters.txt testaccuracy.txt pseudovariables.txt dualvariables.txt rowstoppingparameters.txt pseudostoppingparameters.txt supportvectorscounter.txt primalobjective.txt primalp..arameters.txt trainingtime.txt merging.txt
    
     //dobjfn_dfile_per << "Writing this to a file.\n";
     ofstream dobjfn_dfile_pseudoVariables;
     dobjfn_dfile_pseudoVariables.open (PATH + "pseudovariables.txt");
     //dobjfn_dfile_pseudoVariables << "Writing this to a file.\n";
     ofstream dobjfn_dfile_dualVariables;
     dobjfn_dfile_dualVariables.open (PATH + "dualvariables.txt");
     //dobjfn_dfile_dualVariables << "Writing this to a file.\n";
     //ofstream dobjfn_dfile_pseudoData;
     //dobjfn_dfile_pseudoData.open (PATH + "dpseudoData_dsolver.txt");
     //dobjfn_dfile_pseudoData << "Writing this to a file.\n";
     //ofstream dobjfn_dfile_pseudoLabels;
     //dobjfn_dfile_pseudoLabels.open (PATH + "dpseudoLabels_dsolver.txt");
     //dobjfn_dfile_pseudoLabels << "Writing this to a file.\n";
     ofstream dobjfn_dfile_gradStepdual;
     dobjfn_dfile_gradStepdual.open (PATH + "rowstoppingparameters.txt");
     //dobjfn_dfile_pseudoLabels << "Writing this to a file.\n";
     ofstream dobjfn_dfile_gradSteppseudo;
     dobjfn_dfile_gradSteppseudo.open (PATH + "pseudostoppingparameters.txt");
     //dobjfn_dfile_pseudoLabels << "Writing this to a file.\n";
     ofstream dobjfn_dfile_dualpseudoCounter;
     dobjfn_dfile_dualpseudoCounter.open (PATH + "supportvectorscounter.txt");
    
    ofstream dobjfn_dfile_gradvariants;
    dobjfn_dfile_gradvariants.open (PATH + "gradvariants.txt");
    
    
    ofstream pobjfn_dfile;
    pobjfn_dfile.open (PATH + "primalobjective.txt");
    //pobjfn_dfile << "Writing this to a file.\n";
    ofstream pobjfn_dfile_param;
    pobjfn_dfile_param.open (PATH + "primalparameters.txt");
    //pobjfn_dfile_param << "Writing this to a file.\n";
    
    ofstream dobjfn_dfile_traint;
    dobjfn_dfile_traint.open (PATH + "trainingtime.txt");
    //dobjfn_dfile_traint << "Writing this to a file.\n";
    
    ofstream dobjfn_dfile_merging;
    dobjfn_dfile_merging.open (PATH + "merging.txt");
    //dobjfn_dfile_merging << "Writing this to a file.\n";
    
    double train_start_t = 0.0;
    //double train_end_t = 0.0;
    // bool BreakConditionON = false;
    
    double mergeAndDeleteSV_counter = 0.0;
    unsigned int countMerges = 0;
    
    unsigned int dualvarisZero = 0;
    unsigned int dualvarisC = 0;
    unsigned int dualvarbetZeroC = 0;
    
    INDEX numIter = 1;
    size_t M = 20;                            //200//500 //%4800;           % (EigenPro) subsample size.
    unsigned int topk = 16;                   //160 //10//%160;            % (EigenPro) top-k eigensystem.
    size_t n = number_of_training_points;       //size(train_x, 1); % number of training samples.
    // size_t N = 24;                             //size(train_x, 2); % number of raw features.
    
    for (size_t epoch = 0; epoch < max_epochs; epoch++)
    {
          mergeAndDeleteSV_counter = 0.0;
        double overhead_proj=0.0;
        double Amat_proj = 0.0;
        double B1mat_proj = 0.0;
        double B2mat_proj = 0.0;
        double betamat_proj = 0.0;
        double gamma_proj = 0.0;
        
        
        /*if(epoch == 5 || epoch == 10){// || epoch == 500 || epoch == 1000 || epoch == 1500 || epoch == 2000 || epoch == 2500 ||epoch == 3000 || epoch ==3500 || epoch == 4000 || epoch == 4500){
            pseudo_variables.clear();
            pseudo.data.clear();
            pseudo.labels.clear();
        } */
        //dataset.shuffle_ds_dualvec(dual_variables);
        
        cout << endl << epoch+1 << ":";
        double gradientMIN = INFINITY;
        double gradientMAX = -INFINITY;
        
        double gradientpseudoMIN = INFINITY;
        double gradientpseudoMAX = -INFINITY;
        // double mMIN = -INFINITY;
        double mMIN = INFINITY;
        double mMAX = -INFINITY;
        double pseudo_mMAX = -INFINITY;
        INDEX pseudoGMAX;
        double KL_max = -INFINITY;
        sequence.clear();
        //fillUniformSequence(sequence, number_of_training_points);
        
        double maxPseudoinsequence = -INFINITY;
        train_start_t = (double)clock() / CLOCKS_PER_SEC;
        double maxPseudo = 0.0;
        double maxG_pseudovar = -INFINITY;
        double sum_maxG_pseudovar = 0.0;
        
        unsigned int counter = 0;
        vector<double> extradualvariables;
        sparseData extradataset;
        sequence.clear();
        unsigned int count = 0;
        for (INDEX i=0; i<number_of_training_points; i++)
        {
            sequence.push_back(i);
        }
        for (unsigned int i=0; i<number_of_training_points; i++)
        {
            //swap(ws_vector[i], ws_vector[rand() % ws_vector.size()]);
            swap(sequence[i], sequence[rand() % number_of_training_points]);
        }
        size_t sequence_size = sequence.size();
        int lr = 1;
        
        
        
        std::vector<double> s;
        //MatrixXf V;
        double lambda1;
        double s_sum = 0.0;
        
        double eigenpro_eta = 0.0;
        double conditionNumber ;
        
        for (INDEX i = 0; i < sequence_size; i++)
        {
            numIter++;
            
            //if(numIter % iterFreq == 0)
            //{
            //    train_start_t = (double)clock() / CLOCKS_PER_SEC;
            
            //}
            INDEX ws = sequence[i];
            double alpha_point = dual_variables[ws];
            /*
            //Compute gradient
            //vector<double> gradient_maxPseudo;
            //gradient_maxPseudo =computeGradient(dataset.data[ws], dataset.labels[ws], pseudo_variables, pseudo, kernel);
            
            //double gradient = gradient_maxPseudo[0];
            //maxPseudo =gradient_maxPseudo[1];
            //if(maxPseudo > maxPseudoinsequence)
            //{
              //  maxPseudoinsequence = maxPseudo ;
            //}
            */
            double gradient = computeGradient(dataset.data[ws], dataset.labels[ws], pseudo_variables, pseudo, kernel);
            //vector<double> gradientminmax;
            //gradientminmax = computeGradientPseudoMaxMin(alpha_point, dataset.data[ws], dataset.labels[ws], pseudo_variables, pseudo, kernel);
            
          
            
            //double gradient = gradientminmax[0];
            //double gradientmax = gradientminmax[1];
           // double gradientmin = gradientminmax[2];
           // double pseudogradient   = gradientminmax[3];
            
           // if (pseudogradient < gradientpseudoMIN) gradientpseudoMIN = pseudogradient-(1+gradientmin);
            //if (pseudogradient > gradientpseudoMAX) gradientpseudoMAX = pseudogradient-(1+gradientmax);
            
            //if (gradientmin < gradientMIN) gradientMIN = gradientmin;
            //if (gradientmax > gradientMAX) gradientMAX = gradientmax;
            //if (m_min < mMIN) mMIN = m_min;
            
            double old_alpha = dual_variables[ws];
            double new_alpha = max(0.0, min(old_alpha + gradient, C));
            dual_variables[ws] = new_alpha;
            
            
            
            ///////////////////////////////
            vector<double>p_matrix;
            //p_matrix = computeS_studentDistribution(dataset.data[ws], dataset.labels[ws], pseudo_variables, pseudo, kernel);
            //INDEX gradient_inx = p_matrix.size()-1;
            //double gradient = p_matrix[gradient_inx];
            //cout << gradient << "; " << gradient1 << endl;
            
            
            ////////////////////////////////
            
            
            
            
            //sum of Steps_dualvar computation
            //here 01.05 -->            double oneS_dualvar = std::abs(new_alpha-old_alpha);
            //here 01.05 -->           sum_diffsteps_dualvar += oneS_dualvar;
            
            //max of step_pseudovar computation
            /*here 01.05 -->          if (oneS_dualvar > maxStep_dualvar) maxStep_dualvar = oneS_dualvar;
             
             //maxG computation
             if (old_alpha == 0.0 && gradient <= 0.0){}
             else if (old_alpha == C && gradient >= 0.0){}
             else
             {
             double absG_dualvar = std::abs(gradient);
             if (absG_dualvar > maxG_dualvar) maxG_dualvar = absG_dualvar;
             //sumG_dualvar computation
             sum_maxG_dualvar += absG_dualvar;
             
             }
             */
            
            
            //cout << maxPseudo << endl;
           
            
            if(new_alpha == 0) dualvarisZero+=1;
            if(new_alpha == C) dualvarisC+=1;
            if(new_alpha<C && new_alpha>0) dualvarbetZeroC+=1;
            //////////////////////////////////////////////////////////////////////
            double absG_pseudovar = 0.0;
            int score = 0;
            if (old_alpha == 0.0 && gradient <= 0.0){}
            else if (old_alpha == C && gradient >= 0.0){}
            else
            {
                absG_pseudovar = std::abs(gradient);
                (absG_pseudovar > maxG_pseudovar)? score=1 : score=2 ;
                if (absG_pseudovar > maxG_pseudovar) maxG_pseudovar = absG_pseudovar;
                //sumG_dualvar computation
                sum_maxG_pseudovar += absG_pseudovar;
                
            }
            
            
           
            if ( (new_alpha != old_alpha) )
            //if ((new_alpha != old_alpha) && epoch <= 1 )
            {
                pseudo.data.push_back(dataset.data[ws]);
                pseudo_variables.push_back((new_alpha-old_alpha)); //-old_alpha
                pseudo.labels.push_back(dataset.labels[ws]);
                
            }
            
            
            //////////////////////////////////////////////////////////////////////
 //if (epoch == 1)
   //  cout << "lets see: " << count++  << ":ws: " << ws << endl ;
            if (pseudo.data.size() > B )
            {
                counter++;
                //Check the model size compared to the budget
                double mergeAndDeleteSV_start_time = (double)clock() / CLOCKS_PER_SEC;
               // mergeAndDeleteSV(pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);
                //projectSV(pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);
               //projectAndDeleteSV(pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);
                std::vector<double> project_times = projectSVLinearEquations(dual_variables, dataset, pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);
                overhead_proj+= project_times[0];
                Amat_proj+= project_times[1];
                B1mat_proj+= project_times[2];
                B2mat_proj+= project_times[3];
                betamat_proj+= project_times[4];
                gamma_proj+= project_times[5];
               // multimergeAndDeleteSV(pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);
                double mergeAndDeleteSV_end_time = (double)clock() / CLOCKS_PER_SEC;
                mergeAndDeleteSV_counter += mergeAndDeleteSV_end_time - mergeAndDeleteSV_start_time;
                countMerges++;
                
            }
            /*
            if (pseudo.data.size() > B && epoch >1)
            {
                counter++;
                
                //Check the model size compared to the budget
                double mergeAndDeleteSV_start_time = (double)clock() / CLOCKS_PER_SEC;
                projectSV(pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);
                double mergeAndDeleteSV_end_time = (double)clock() / CLOCKS_PER_SEC;
                mergeAndDeleteSV_counter += mergeAndDeleteSV_end_time - mergeAndDeleteSV_start_time;
                countMerges++;
                
            } */
            
        
           /* if (pseudo.data.size() > B)
            {
                switch (epoch )
                    
                {
                        case 0:
                    {
                    //counter++;
                    
                    //Check the model size compared to the budget
                    double mergeAndDeleteSV_start_time = (double)clock() / CLOCKS_PER_SEC;
                    mergeAndDeleteSV(pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);
                    
                    double mergeAndDeleteSV_end_time = (double)clock() / CLOCKS_PER_SEC;
                    mergeAndDeleteSV_counter += mergeAndDeleteSV_end_time - mergeAndDeleteSV_start_time;
                        countMerges++;
                        break;
                        
                    }   case 1:
                    {
                        //counter++;
                        
                        //Check the model size compared to the budget
                        double mergeAndDeleteSV_start_time = (double)clock() / CLOCKS_PER_SEC;
                        mergeAndDeleteSV(pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);
                        
                        double mergeAndDeleteSV_end_time = (double)clock() / CLOCKS_PER_SEC;
                        mergeAndDeleteSV_counter += mergeAndDeleteSV_end_time - mergeAndDeleteSV_start_time;
                        countMerges++;
                        break;
                        
                    }
                    default:
                        break;
                        
                }
                
            }*/

            //////////////////////////////////////////////////////////////////////
            // here 01.05 -->
            // if(numIter%iterFreq == 0 )
             
            // {
               //  train_end_t = (double)clock() / CLOCKS_PER_SEC;
              //   dobjfn_dfile_traint << "epoch :"  << epoch +1  << ":trainingtime.:"  << train_end_t - train_start_t << ":" << endl;
               //  train_start_t = 0.0;
                // train_end_t = 0.0;
             
             // dobjfn_dfile_traint << "epoch :"  << epoch +1  << ":trainingtime.:"  << train_end_t - train_start_t << ":" << endl;
             // train_start_t = 0.0;
             
             // dualObjFunValue =  \sum dual-variables - 0.5 (minW)^2 //
             
                // std::tie(dualObjFunValue, dualVariable, dual_05_minWsquare) = dualObjectiveFunction ( C, pseudo_variables, dual_variables, pseudo, dataset, kernel );
               //  dobjfn_dfile <<"epoch:"  << epoch +1 << ":" << dualObjFunValue << ":" << endl;
               //  dobjfn_dfile_param << "epoch:"  <<epoch +1 <<  ":sumOfalpha:" << dualVariable << ":(1/2)minWsquare:" << dual_05_minWsquare << ":"  << endl;
             
             ///////////////////////////////////////////////////////////
             // primalObjFunValue =  C*HLoss + 0.5 (minW)^2 //
             
           //  std::tie(primalObjFunValue, primal_05_minWsquare, primal_C_mul_Hloss) = primalObjectiveFunction (C, pseudo_variables, dual_variables, pseudo, dataset, kernel );
            // pobjfn_dfile <<"epoch:"  <<epoch +1 << ":" << primalObjFunValue << ":" << endl;
           //  pobjfn_dfile_param << "epoch:"  <<epoch +1  << ":C*HLoss:" << primal_C_mul_Hloss << ":(1/2)minWsquare:" << primal_05_minWsquare << ":" << endl ;
             //}
             ///////////////////////////////////////////////////////////
             // save pseudo-variables //
             /*
             dobjfn_dfile_pseudoVariables << "epoch:"  << epoch +1 << ":" ;
             for(int pseudo_index = 0; pseudo_index < pseudo_variables.size(); pseudo_index++ )
             {
             dobjfn_dfile_pseudoVariables   << pseudo_variables[pseudo_index] << ":";
             }
             dobjfn_dfile_pseudoVariables << endl;
             
             //////////////////////////////////////////////////////////
             // save pseudo-data //
             
             dobjfn_dfile_pseudoData << "epoch:"  << epoch +1 << ":" ;
             for(int pseudo_index = 0; pseudo_index < pseudo.data.size(); pseudo_index++ )
             {
             vector<SE> first_vector = pseudo.data[pseudo_index];
             size_t x1_size = first_vector.size();
             
             for (size_t i = 0; i< x1_size-1; i++){dobjfn_dfile_pseudoData  << first_vector[i].index <<";"<< first_vector[i].value << ":";}
             dobjfn_dfile_pseudoData  << endl;
             dobjfn_dfile_pseudoData << " : :";
             
             }
             dobjfn_dfile_pseudoData << endl;
             
             //////////////////////////////////////////////////////////
             // save pseudo-labels //
             
             dobjfn_dfile_pseudoLabels << "epoch:"  << epoch +1 << ":" ;
             for(int pseudo_index = 0; pseudo_index < pseudo_variables.size(); pseudo_index++ )
             {
             dobjfn_dfile_pseudoLabels   << (int)pseudo.labels[pseudo_index] << ":";
             }
             dobjfn_dfile_pseudoLabels << endl;
             
             //////////////////////////////////////////////////////////
             // save dual-variables //
             
             dobjfn_dfile_dualVariables << "epoch:"  << epoch +1 << ":" ;
             for(int dual_index = 0; dual_index < dual_variables.size(); dual_index++ )
             {
             dobjfn_dfile_dualVariables   << dual_variables[dual_index] << ":";
             }
             dobjfn_dfile_dualVariables << endl;
             
             
             
             
             }//end of 5000 iter.
             */ // here 01.05 -->
            
            
            //////////////////////////////////////////////////////////
           // if(numIter%iterFreq == 0 )
                
           // {
                
                //////////////////////////////////////////////////////////
                /* merging time, training time and performance at each epoch */
               // dobjfn_dfile_merging << "epoch :"  << epoch +1  << ":mergingtime:" << mergeAndDeleteSV_counter << ":mergingsteps:" << countMerges << endl;
               // countMerges = 0;
              ///  mergeAndDeleteSV_counter = 0.0;
                
              //  SVM svm(pseudo_variables, pseudo, kernel);
              //  dobjfn_dfile_per << "epoch :"  << epoch +1  << ": per.:"  << svm.evaluateTestset(testdataset)<< ":" << endl;
                //////////////////////////////////////////////////////////////////////
                /* gradient (max, sum, avg), step (max, sum, avg) --> dual variables*/
                /*  dobjfn_dfile_gradStepdual << "epoch :"  << epoch +1  << ":sum_diffsteps_dualvar:" << sum_diffsteps_dualvar << ":avg_diffsteps_dualvar:" << sum_diffsteps_dualvar/iterFreq  << ":maxG_dualvar:" <<maxG_dualvar<< ":sum_maxG_dualvar:"  <<sum_maxG_dualvar << ":avg_maxG_dualvar:"  <<sum_maxG_dualvar/iterFreq  << ":maxStep_dualvar:" << maxStep_dualvar<<":" << endl;
                 sum_diffsteps_dualvar=0.0;
                 maxG_dualvar=0.0;
                 sum_maxG_dualvar=0.0;
                 maxG_dualvar=0.0;
                 ////////////////////////////////////////////////////////////////////////
                 //gradient (max, sum, avg), step (max, sum, avg) --> pseudo variables
                 dobjfn_dfile_gradSteppseudo << "epoch :"  << epoch +1  << ":sum_diffsteps_pseudovar:" << sum_diffsteps_pseudovar << ":avg_diffsteps_pseudovar:" << sum_diffsteps_pseudovar/iterFreq  << ":maxG_pseudovar:"<<maxG_pseudovar<< ":sum_maxG_pseudovar:"  <<sum_maxG_pseudovar<< ":maxStep_pseudovar:" << maxStep_pseudovar<< ":avg_maxG_pseudovar:"  <<sum_maxG_pseudovar/iterFreq << ":" << endl;
                 
                 sum_diffsteps_pseudovar=0.0;
                 maxG_pseudovar=0.0;
                 sum_maxG_pseudovar=0.0;
                 maxG_pseudovar=0.0;
                 */
               // train_start_t = (double)clock() / CLOCKS_PER_SEC;
                
                //////////////////////////////////////////////////////////
          //  }
            
            
        } //end of sequence
       
        /*std::tie(s, V, lambda1)=  rsvd ( pseudo, kernel, M,  topk);
        eigenpro_eta = 1.5 / (2 * s[0] / n);
        conditionNumber = s[0]/s[10];
        dobjfn_dfile_pseudoVariables <<"epoch :"  << epoch +1  << ":eigenpro_eta:"  << eigenpro_eta<< ":conditionNumber:" << conditionNumber << ":lambda1:"<< lambda1 << endl;
      */
        double train_end_t = (double)clock() / CLOCKS_PER_SEC;
        //dobjfn_dfile_merging << "epoch :"  << epoch +1 << ":KL:" << KL_max;
     
        dobjfn_dfile_merging << "epoch :"  << epoch +1  <<  ":mergingtime:" << mergeAndDeleteSV_counter << ":mergingsteps:" << countMerges  << ":overhead_proj:" << overhead_proj << ":Amat_proj:" << Amat_proj << ":B1mat_proj:" << B1mat_proj << ":B2mat_proj:" << B2mat_proj << ":betamat_proj:" << betamat_proj << ":gamma_proj:" << gamma_proj <<  endl;
        
        dobjfn_dfile_gradvariants << "epoch :"  << epoch +1  << ":maxG_pseudovar:"  << maxG_pseudovar<< ":avg gradient:" << maxG_pseudovar/sum_maxG_pseudovar << ":minPgradient:" << gradientMIN << ":maxPgradient:" << gradientMAX <<  ":gradientpseudoMIN:" << gradientpseudoMIN << ":gradientpseudoMAX:" << gradientpseudoMAX  << endl;
        //dobjfn_dfile_dualpseudoCounter << "epoch :"  << epoch +1  <<
        /*pseudo_variables[pseudoGMAX] = dual_variables.back();
        pseudo_variables.pop_back();
        
        pseudo.data[pseudoGMAX] = dataset.data.back();
        pseudo.data.pop_back();
        
        pseudo.labels[pseudoGMAX] = dataset.labels.back();
        pseudo.labels.pop_back();*/
        
        countMerges = 0;
        sum_maxG_pseudovar = 0.0;
        maxG_pseudovar = -INFINITY;
        
        SVM svm(pseudo_variables, pseudo, kernel);
        dobjfn_dfile_per << "epoch :"  << epoch +1  << ": per.:"  << svm.evaluateTestset(testdataset)<< ":" << endl;
        unsigned int nSV = 0; unsigned int nBSV = 0;
        for (size_t i=0; i<number_of_training_points; i++)
        {
            if(fabs(dual_variables[i])>0)
            {
                ++nSV;
                if(dataset.labels[i]>0)
                {
                    if(fabs(dual_variables[i])>=C)++nBSV;
                    
                } else
                {
                    if(fabs(dual_variables[i])>=0)++nBSV;
                }
            }
            
        }
        dobjfn_dfile_dualpseudoCounter <<"epoch :"  << epoch +1 << ":nSV: "<<nSV<<":nBSV:"<<nBSV<< endl;
        /*for (int i= 0; i < pseudo_variables.size(); i++)
        {
            
            extradualvariables.push_back(pseudo_variables[i]);
            extradataset.data.push_back(pseudo.data[i]);
            extradataset.labels.push_back(pseudo.labels[i]);
        }*/
       
        
        dobjfn_dfile_traint << "epoch :"  << epoch +1  << ":trainingtime.:"  << train_end_t - train_start_t << ":" << endl;
       // dobjfn_dfile_dualpseudoCounter << "epoch :"  << epoch +1  << ":maxG:" << gradientMAX << ":minG:" << gradientMIN <<  ":dualvarisZero.:"  << dualvarisZero << ":dualvarisC.:"  << dualvarisC << ":dualvarbetZeroC.:"  << dualvarbetZeroC<< ":" << endl;
        //if (dualvarbetZeroC <= number_of_training_points/3)
        //{break;}
        
        /* dualObjFunValue =  \sum dual-variables - 0.5 (minW)^2 */
        double dualObjFunValue = 0.0;
        double dualVariable = 0.0;
        double dual_05_minWsquare = 0.0;
        std::tie(dualObjFunValue, dualVariable, dual_05_minWsquare) = dualObjectiveFunction ( C, pseudo_variables, dual_variables, pseudo, dataset, kernel );
          dobjfn_dfile <<"epoch:"  << epoch +1 << ":" << dualObjFunValue << ":" << endl;
         dobjfn_dfile_param << "epoch:"  <<epoch +1  << ":sumOfalpha:" << dualVariable << ":(1/2)minWsquare:" << dual_05_minWsquare << ":"  << endl;
        dualObjFunValue = 0.0;
        dualVariable = 0.0;
        dual_05_minWsquare = 0.0;
        /* primalObjFunValue =  C*HLoss + 0.5 (minW)^2 */
        double primalObjFunValue = 0.0;
        double primal_05_minWsquare = 0.0;
        double primal_C_mul_Hloss = 0.0;
        
                   std::tie(primalObjFunValue, primal_05_minWsquare, primal_C_mul_Hloss) = primalObjectiveFunction (C, pseudo_variables, dual_variables, pseudo, dataset, kernel );
        pobjfn_dfile <<"epoch:"  <<epoch +1 << ":" << primalObjFunValue << ":" << endl;
        pobjfn_dfile_param << "epoch:"  <<epoch +1  << ":C*HLoss:" << primal_C_mul_Hloss << ":(1/2)minWsquare:" << primal_05_minWsquare << ":" << endl ;
        
        primalObjFunValue = 0.0;
        primal_05_minWsquare = 0.0;
        primal_C_mul_Hloss = 0.0;
        
        dualvarisZero  = 0;
        dualvarisC     = 0;
        dualvarbetZeroC = 0;
        //if(epoch == 1 ||epoch == 9 || epoch ==19  || epoch ==39 || epoch == 49 )
        //{
        /* dualObjFunValue =  \sum dual-variables - 0.5 (minW)^2 */
                   //std::tie(dualObjFunValue, dualVariable, dual_05_minWsquare) = dualObjectiveFunction ( C, pseudo_variables, dual_variables, pseudo, dataset, kernel );
        //  dobjfn_dfile <<"epoch:"  << epoch +1 << ":" << dualObjFunValue << ":" << endl;
        // dobjfn_dfile_param << "epoch:"  <<epoch +1  << ":sumOfalpha:" << dualVariable << ":(1/2)minWsquare:" << dual_05_minWsquare << ":"  << endl;
        
        /* primalObjFunValue =  C*HLoss + 0.5 (minW)^2 */
          //            std::tie(primalObjFunValue, primal_05_minWsquare, primal_C_mul_Hloss) = primalObjectiveFunction (C, pseudo_variables, dual_variables, pseudo, dataset, kernel );
         //pobjfn_dfile <<"epoch:"  <<epoch +1 << ":" << primalObjFunValue << ":" << endl;
         //pobjfn_dfile_param << "epoch:"  <<epoch +1  << ":C*HLoss:" << primal_C_mul_Hloss << ":(1/2)minWsquare:" << primal_05_minWsquare << ":" << endl ;
        
        /* save pseudo-variables */
        // dobjfn_dfile_pseudoVariables << "epoch:"  << epoch +1 << ":" ;
        // for(int pseudo_index = 0; pseudo_index < pseudo_variables.size(); pseudo_index++ )
        //{
          //  dobjfn_dfile_pseudoVariables   << pseudo_variables[pseudo_index] << ":";
        // }
        //dobjfn_dfile_pseudoVariables << endl;
        
        /* save pseudo-data */
        /*               dobjfn_dfile_pseudoData << "epoch:"  << epoch +1 << ":" ;
         for(int pseudo_index = 0; pseudo_index < pseudo.data.size(); pseudo_index++ )
         {
         vector<SE> first_vector = pseudo.data[pseudo_index];
         size_t x1_size = first_vector.size();
         
         for (size_t i = 0; i< x1_size-1; i++){dobjfn_dfile_pseudoData  << first_vector[i].index <<";"<< first_vector[i].value << ":";}
         dobjfn_dfile_pseudoData  << endl;
         dobjfn_dfile_pseudoData << " : :";
         
         }
         dobjfn_dfile_pseudoData << endl;
         */
        /* save pseudo-labels */
        /*           dobjfn_dfile_pseudoLabels << "epoch:"  << epoch +1 << ":" ;
         for(int pseudo_index = 0; pseudo_index < pseudo_variables.size(); pseudo_index++ )
         {
         dobjfn_dfile_pseudoLabels   << (int)pseudo.labels[pseudo_index] << ":";
         }
         dobjfn_dfile_pseudoLabels << endl;
         */
        /* save dual-variables */
       // dobjfn_dfile_dualVariables << "epoch:"  << epoch +1 << ":" ;
        //for(int dual_index = 0; dual_index < dual_variables.size(); dual_index++ )
        //{
          //  dobjfn_dfile_dualVariables   << dual_variables[dual_index] << ":";
        //}
        //dobjfn_dfile_dualVariables << endl;
        
        
        /* merging time, training time and performance at each epoch */
        //dobjfn_dfile_merging << "epoch :"  << epoch +1  << ":mergingtime:" << mergeAndDeleteSV_counter << ":mergingsteps:" << countMerges << endl;
        //countMerges = 0;
        
        //SVM svm(pseudo_variables, pseudo, kernel);
        //dobjfn_dfile_per << "epoch :"  << epoch +1  << ": per.:"  << svm.evaluateTestset(testdataset)<< ":" << endl;
        
        // dobjfn_dfile_traint << "epoch :"  << epoch +1  << ":trainingtime.:"  << train_end_t - train_start_t << ":" << endl;
        
        /*if(epoch == 1)avgStepInitial = avgStep;
         
         if( (avgStep < 0.95 * minAVGgain) &&  avgStep/avgStepInitial < 0.95 )
         //std::abs(avgG)<0.95*minAVGgain &&
         {
         
         minAVGgain = avgStep;
         count_maxG = 0;
         //cout << "avgG:" << avgG << "minAVGgain:" << std::abs(minAVGgain)<< endl;
         }
         else{
         count_maxG++;
         //cout << "count_maxG:" << count_maxG;
         //cout << "avgG:" << std::abs(avgStep) << endl;
         if(count_maxG > 3)break;
         }*/
        //dobjfn_dfile_dualpseudoCounter << "epoch :"  << epoch +1  << ":dualvarisZero.:"  << dualvarisZero << ":dualvarisC.:"  << dualvarisC << ":dualvarbetZeroC.:"  << dualvarbetZeroC<< ":" << endl;
        //dualvarisZero  = 0;
        //dualvarisC     = 0;
        //dualvarbetZeroC = 0;
    //}
        
    }
    dobjfn_dfile.close();
    pobjfn_dfile.close();
    
    dobjfn_dfile_param.close();
    pobjfn_dfile_param.close();
    
    dobjfn_dfile_per.close();
    dobjfn_dfile_traint.close();
    dobjfn_dfile_merging.close();
    
    dobjfn_dfile_gradvariants.close();
    
    dobjfn_dfile_pseudoVariables.close();
    //dobjfn_dfile_pseudoData.close();
    //dobjfn_dfile_pseudoLabels.close();
    //dobjfn_dfile_dualVariables.close();
    
    //dobjfn_dfile_gradSteppseudo.close();
    //dobjfn_dfile_gradStepdual.close();
    //dobjfn_dfile_dualpseudoCounter.close();
    
    return SVM(pseudo_variables, pseudo, kernel);
}



SVM BMVPSMO(sparseData& dataset, sparseData& testdataset, double C, Kernel& kernel, LookupTable& wd_parameters, double accuracy, size_t B, size_t max_epochs, Heuristic heuristic)
{
    cout << "Optimisation ... begin \n";
    size_t number_of_training_points = dataset.data.size();
    cout << "Number of training points: " << number_of_training_points << endl;
    // unsigned int iterFreq = 100000;
    sparseData pseudo;
    vector<double> pseudo_variables;
    
    vector<INDEX> sequence(0);
    vector<double> dual_variables(number_of_training_points, 0);
    
    // Main optimization loop
    //double dualObjFunValue = 0.0 , primalObjFunValue = 0.0;
    //double dualVariable = 0.0, dual_05_minWsquare = 0.0;
    //double primal_05_minWsquare = 0.0 , primal_C_mul_Hloss = 0.0;
    
    std::string PATH = "Exp1_SMO_B50/";
    system("mkdir \"Exp1_SMO_B50\"");
    cout << "Max Epochs: " << max_epochs << "  Current:";
    ofstream dobjfn_dfile;
    dobjfn_dfile.open (PATH + "dualobjective.txt");
    //dobjfn_dfile << "Writing this to a file.\n";
    ofstream dobjfn_dfile_param;
    dobjfn_dfile_param.open (PATH + "dualparameters.txt");
    //dobjfn_dfile_param << "Writing this to a file.\n";
    ofstream dobjfn_dfile_per;
    dobjfn_dfile_per.open (PATH + "testaccuracy.txt");//dualparameters.txt testaccuracy.txt pseudovariables.txt dualvariables.txt rowstoppingparameters.txt pseudostoppingparameters.txt supportvectorscounter.txt primalobjective.txt primalparameters.txt trainingtime.txt merging.txt
    
    //dobjfn_dfile_per << "Writing this to a file.\n";
    ofstream dobjfn_dfile_pseudoVariables;
    dobjfn_dfile_pseudoVariables.open (PATH + "pseudovariables.txt");
    //dobjfn_dfile_pseudoVariables << "Writing this to a file.\n";
    ofstream dobjfn_dfile_dualVariables;
    dobjfn_dfile_dualVariables.open (PATH + "dualvariables.txt");
    //dobjfn_dfile_dualVariables << "Writing this to a file.\n";
    //ofstream dobjfn_dfile_pseudoData;
    //dobjfn_dfile_pseudoData.open (PATH + "dpseudoData_dsolver.txt");
    //dobjfn_dfile_pseudoData << "Writing this to a file.\n";
    //ofstream dobjfn_dfile_pseudoLabels;
    //dobjfn_dfile_pseudoLabels.open (PATH + "dpseudoLabels_dsolver.txt");
    //dobjfn_dfile_pseudoLabels << "Writing this to a file.\n";
    ofstream dobjfn_dfile_gradStepdual;
    dobjfn_dfile_gradStepdual.open (PATH + "rowstoppingparameters.txt");
    //dobjfn_dfile_pseudoLabels << "Writing this to a file.\n";
    ofstream dobjfn_dfile_gradSteppseudo;
    dobjfn_dfile_gradSteppseudo.open (PATH + "pseudostoppingparameters.txt");
    //dobjfn_dfile_pseudoLabels << "Writing this to a file.\n";
    ofstream dobjfn_dfile_dualpseudoCounter;
    dobjfn_dfile_dualpseudoCounter.open (PATH + "supportvectorscounter.txt");
    
    
    ofstream pobjfn_dfile;
    pobjfn_dfile.open (PATH + "primalobjective.txt");
    //pobjfn_dfile << "Writing this to a file.\n";
    ofstream pobjfn_dfile_param;
    pobjfn_dfile_param.open (PATH + "primalparameters.txt");
    //pobjfn_dfile_param << "Writing this to a file.\n";
    
    ofstream dobjfn_dfile_traint;
    dobjfn_dfile_traint.open (PATH + "trainingtime.txt");
    //dobjfn_dfile_traint << "Writing this to a file.\n";
    
    ofstream dobjfn_dfile_merging;
    dobjfn_dfile_merging.open (PATH + "merging.txt");
    //dobjfn_dfile_merging << "Writing this to a file.\n";
    
    double train_start_t = 0.0;
    //double train_end_t = 0.0;
    // bool BreakConditionON = false;
    
    double mergeAndDeleteSV_counter = 0.0;
    unsigned int countMerges = 0;
    //double sum_diffsteps_dualvar = 0.0;
    //double sum_diffsteps_pseudovar = 0.0;
    //double sum_maxG_dualvar = 0.0;
    //double sum_maxG_pseudovar = 0.0;
    
    //double maxG_dualvar = -INFINITY;
    //double maxG_pseudovar = -INFINITY;
    //double maxStep_dualvar = -INFINITY;
    //double maxStep_pseudovar = -INFINITY;
    
    unsigned int dualvarisZero = 0;
    unsigned int dualvarisC = 0;
    unsigned int dualvarbetZeroC = 0;
    

    //INDEX ws;

    //size_t number_of_support_vectors = 0;
    vector<double> gradients(number_of_training_points, 1.0);
    

    //size_t iteration_counter = 0;
    // Initialize constraints for y*alpha
    vector<double> lower_constraints_combined(number_of_training_points, 0);
    vector<double> upper_constraints_combined(number_of_training_points, 0);
    for (INDEX i = 0; i < number_of_training_points; i++) {
        if (dataset.labels[i] == 1) {
            lower_constraints_combined[i] = 0;
            upper_constraints_combined[i] = C;
        } else if (dataset.labels[i] == -1) {
            lower_constraints_combined[i] = -C;
            upper_constraints_combined[i] = 0;
        }
    }
    
    vector<double>ygi_storage;
    vector<double>ygj_storage;
     //int counterij = 0;
    for(int ii=0;ii<max_epochs; ii++)
   // while (true)
    {
        tuple<INDEX, INDEX, bool> working_set = findMVP(dataset.labels, gradients, dual_variables, lower_constraints_combined, upper_constraints_combined, accuracy);
        INDEX i, j, ws;
        i = get<0>(working_set);
        j = get<1>(working_set);
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
        //Optimality criterion (relaxed) (instead of yg_i <= yg_j we use yg_i - yg_j < epsilon like LIBSVM)
        //eps is 0.001 in LIBSVM, cf. http://www.csie.ntu.edu.tw/~r94100/libsvm-2.8/README
      
       
        ygi_storage.push_back(yg_i);
        ygj_storage.push_back(yg_j);
        cout << std::abs(yg_i) << ":"<<std::abs(yg_j)<<endl;
        if (gradients[i] < gradients[j])
        //if (abs(yg_j) < abs(yg_i))
        {
            ws = i;
        }
        else {
            ws = j;
        }
        
        
        // Compute gradient!
        double gradient = computeGradient(dataset.data[ws], dataset.labels[ws], pseudo_variables, pseudo, kernel);
        
        // Optimize dual objective function over the chosen working set (direction search / newton step)
        double newton_max = gradient;
        
        // Truncate optimum (dual objective function is concave, so naive truncation makes sense)
        double old_alpha = dual_variables[ws];
        double new_alpha = max(0.0, min(old_alpha + newton_max, C));
        
        /*
        counterij++;
        if(counterij > 4 && pseudo.data.size() > (B-1) )
        {
            
            size_t ygi_storage_size = ygi_storage.size()-1;
            size_t ygj_storage_size = ygj_storage.size()-1;
            if(ygi_storage[ygi_storage_size] + ygi_storage[ygi_storage_size-1] - 2*ygi_storage[ygi_storage_size-2]<0.0000001)
            {
                if(ygj_storage[ygj_storage_size] + ygj_storage[ygj_storage_size-1] - 2*ygj_storage[ygj_storage_size-2]<0.0000001)
                {
                    cout << "now it should break\n";
                    break;
                }
            }
        }
        */
        //Direction Search
       // double newton_min = (yg_i - yg_j)/(kernel.evaluate(dataset.data[i], dataset.data[i]) + kernel.evaluate( dataset.data[j], dataset.data[j]) - 2*kernel.evaluate(dataset.data[i],  dataset.data[j]));
       // double lambda = min(B_i - ya_i, min(ya_j - A_j, newton_min));
        // Gradient Update
       // for (INDEX index = 0; index < gradients.size(); index++) {
         //   double gradient_change = lambda*dataset.labels[index]*(kernel.evaluate(dataset.data[j], dataset.data[index]) - kernel.evaluate(dataset.data[i], dataset.data[index]));
          //  gradients[index] += gradient_change;
       // }
        
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
        
        
        //Dual variable update
        dual_variables[ws] = new_alpha;
        
        // Keep track of SVs
       // if ((old_alpha == 0) && (new_alpha != 0)) {
            if ((old_alpha - new_alpha != 0)) {

        
        // Maintain Pseudorepresentation
        pseudo.data.push_back(dataset.data[ws]);
        pseudo_variables.push_back(new_alpha - old_alpha );
        pseudo.labels.push_back(dataset.labels[ws]);
        
      // pseudo.data.push_back(dataset.data[j]);
      // pseudo_variables.push_back(new_alpha_j - old_alpha_j );
      // pseudo.labels.push_back(dataset.labels[j]);
        
        }
   
    bool gradient_change = false;
        while (pseudo.data.size() > B)
        {
            
            //Check the model size compared to the budget
            double mergeAndDeleteSV_start_time = (double)clock() / CLOCKS_PER_SEC;
            mergeAndDeleteSV(pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);
            //DeleteSV(pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);
            
            
            double mergeAndDeleteSV_end_time = (double)clock() / CLOCKS_PER_SEC;
            mergeAndDeleteSV_counter += mergeAndDeleteSV_end_time - mergeAndDeleteSV_start_time;
            countMerges++;
            gradient_change = true;
            
        }
        
        for (INDEX index = 0; index < gradients.size(); index++)
        {
            //gradients[index] -= dataset.labels[ws]*dataset.labels[index]*kernel.evaluate(dataset.data[index], dataset.data[ws])*(new_alpha - old_alpha);
            //if (gradient_change) gradients[index] -= dataset.labels[ws]*dataset.labels[index]*kernel.evaluate(dataset.data[index], dataset.data[ws])*dual_variables[number_of_training_points-1];
            gradients[index] = computeGradient(dataset.data[index], dataset.labels[index], pseudo_variables, pseudo, kernel);
        }
    // Compute approximate gradients for i & j
        //gradients[ws] = computeGradient(dataset.data[ws], dataset.labels[ws], pseudo_variables, pseudo, kernel);
        //gradients[j] = computeGradient(dataset.data[j], dataset.labels[j], pseudo_variables, pseudo, kernel);
    
    //dual_variables[ws] = new_alpha;
    //dual_variables[j] = new_alpha;
    
     }
    double train_end_t = (double)clock() / CLOCKS_PER_SEC;
    //dobjfn_dfile_merging << "epoch :"  << epoch +1 << ":KL:" << KL_max;
   // dobjfn_dfile_merging << "epoch :"  << epoch +1  << ":KL:"  << KL_max<< ":maxPseudogradient:" << maxPseudo << ":mergingtime:" << mergeAndDeleteSV_counter << ":mergingsteps:" << countMerges << endl;
    countMerges = 0;
    
    SVM svm(pseudo_variables, pseudo, kernel);
   // double svmEvaluate = svm.evaluateTestset(testdataset);
    
    dobjfn_dfile_per   << ": per.:"  << svm.evaluateTestset(testdataset)<< ":" << endl;
    
    dobjfn_dfile_traint  << ":trainingtime.:"  << train_end_t - train_start_t << ":" << endl;
    dobjfn_dfile_dualpseudoCounter  << ":dualvarisZero.:"  << dualvarisZero << ":dualvarisC.:"  << dualvarisC << ":dualvarbetZeroC.:"  << dualvarbetZeroC<< ":" << endl;
    dualvarisZero  = 0;
    dualvarisC     = 0;
    dualvarbetZeroC = 0;
    

dobjfn_dfile.close();
pobjfn_dfile.close();

dobjfn_dfile_param.close();
pobjfn_dfile_param.close();

dobjfn_dfile_per.close();
dobjfn_dfile_traint.close();
dobjfn_dfile_merging.close();


return SVM(pseudo_variables, pseudo, kernel);
}

//Budgeted dual solver
//Budgeted dual solver
SVM acfBDCA(sparseData& dataset, sparseData& testdataset, double C, Kernel& kernel, LookupTable& wd_parameters, double accuracy, size_t B, size_t max_epochs, Heuristic heuristic)
{
    cout << "Optimisation ... begin \n";
    size_t number_of_training_points = dataset.data.size();
    //int l = number_of_training_points; //to be removed
    cout << "Number of training points: " << number_of_training_points << endl;
    // unsigned int iterFreq = 100000;
    sparseData pseudo;
    vector<double> pseudo_variables;
    
    
    vector<INDEX> sequence(0);
    vector<double> dual_variables(number_of_training_points, 0);
    vector<INDEX> ws_sequence(0);
    
    // Main optimization loop
    
    double dualObjFunValue = 0.0 , primalObjFunValue = 0.0, dualobjfnt= 0.0;
    double dualVariable = 0.0, dual_05_minWsquare = 0.0;
    double primal_05_minWsquare = 0.0 , primal_C_mul_Hloss = 0.0;
    
    
    std::string PATH = "s_acf_merge500_v4/";
    system("mkdir \"s_acf_merge500_v4\"");
    cout << "Max Epochs: " << max_epochs << "  Current:";
    ofstream dobjfn_dfile;
    dobjfn_dfile.open (PATH + "dualobjective.txt");
    //dobjfn_dfile << "Writing this to a file.\n";
    ofstream dobjfn_dfile_param;
    dobjfn_dfile_param.open (PATH + "dualparameters.txt");
    //dobjfn_dfile_param << "Writing this to a file.\n";
    ofstream dobjfn_dfile_per;
    dobjfn_dfile_per.open (PATH + "testaccuracy.txt");//dualparameters.txt testaccuracy.txt pseudovariables.txt dualvariables.txt rowstoppingparameters.txt pseudostoppingparameters.txt supportvectorscounter.txt primalobjective.txt primalparameters.txt trainingtime.txt merging.txt
    
    //dobjfn_dfile_per << "Writing this to a file.\n";
    ofstream dobjfn_dfile_pseudoVariables;
    dobjfn_dfile_pseudoVariables.open (PATH + "pseudovariables.txt");
    //dobjfn_dfile_pseudoVariables << "Writing this to a file.\n";
    ofstream dobjfn_dfile_dualVariables;
    dobjfn_dfile_dualVariables.open (PATH + "dualvariables.txt");
    //dobjfn_dfile_dualVariables << "Writing this to a file.\n";
    //ofstream dobjfn_dfile_pseudoData;
    //dobjfn_dfile_pseudoData.open (PATH + "dpseudoData_dsolver.txt");
    //dobjfn_dfile_pseudoData << "Writing this to a file.\n";
    //ofstream dobjfn_dfile_pseudoLabels;
    //dobjfn_dfile_pseudoLabels.open (PATH + "dpseudoLabels_dsolver.txt");
    //dobjfn_dfile_pseudoLabels << "Writing this to a file.\n";
    ofstream dobjfn_dfile_gradStepdual;
    dobjfn_dfile_gradStepdual.open (PATH + "rowstoppingparameters.txt");
    //dobjfn_dfile_pseudoLabels << "Writing this to a file.\n";
    ofstream dobjfn_dfile_gradSteppseudo;
    dobjfn_dfile_gradSteppseudo.open (PATH + "pseudostoppingparameters.txt");
    //dobjfn_dfile_pseudoLabels << "Writing this to a file.\n";
    ofstream dobjfn_dfile_dualpseudoCounter;
    dobjfn_dfile_dualpseudoCounter.open (PATH + "supportvectorscounter.txt");
    ofstream dobjfn_dfile_gradvariants;
    dobjfn_dfile_gradvariants.open (PATH + "gradvariants.txt");
    
    ofstream pobjfn_dfile;
    pobjfn_dfile.open (PATH + "primalobjective.txt");
    //pobjfn_dfile << "Writing this to a file.\n";
    ofstream pobjfn_dfile_param;
    pobjfn_dfile_param.open (PATH + "primalparameters.txt");
    //pobjfn_dfile_param << "Writing this to a file.\n";
    
    ofstream dobjfn_dfile_traint;
    dobjfn_dfile_traint.open (PATH + "trainingtime.txt");
    //dobjfn_dfile_traint << "Writing this to a file.\n";
    
    ofstream dobjfn_dfile_merging;
    dobjfn_dfile_merging.open (PATH + "merging.txt");
    //dobjfn_dfile_merging << "Writing this to a file.\n";
    
    ofstream dfile_wsSize;
    dfile_wsSize.open (PATH + "wsSizeinACF.txt");
    
    ofstream  dobjfn_dfile_maxwd;
    dobjfn_dfile_maxwd.open (PATH +"maxwd.txt");
    
    ofstream  dobjfn_dfile_maxG;
    dobjfn_dfile_maxG.open (PATH +"maxG.txt");
    
    ofstream  dobjfn_dfile_minG;
    dobjfn_dfile_minG.open (PATH +"minG.txt");
    
    ofstream  dobjfn_dfile_maxpseudoG;
    dobjfn_dfile_maxpseudoG.open (PATH +"maxpseudoG.txt");
    
    ofstream  dobjfn_dfile_minpseudoG;
    dobjfn_dfile_minpseudoG.open (PATH +"minpseudoG.txt");
    
    
    double train_start_t = 0.0;
    //double train_end_t = 0.0;
    // bool BreakConditionON = false;
    
    double mergeAndDeleteSV_counter = 0.0;
    unsigned int countMerges = 0;
    //double sum_diffsteps_dualvar = 0.0;
    //double sum_diffsteps_pseudovar = 0.0;
    //double sum_maxG_dualvar = 0.0;
    double sum_maxG_pseudovar = 0.0;
    
    //double maxG_dualvar = -INFINITY;
    double maxG_pseudovar = -INFINITY;
    //double maxStep_dualvar = -INFINITY;
    //double maxStep_pseudovar = -INFINITY;
    //cout << endl << epoch+1 << ":";
    double gradientMIN = INFINITY;
    double gradientMAX = -INFINITY;
    
    double gradientpseudoMIN = INFINITY;
    double gradientpseudoMAX = -INFINITY;
    unsigned int dualvarisZero = 0;
    unsigned int dualvarisC = 0;
    unsigned int dualvarbetZeroC = 0;
    //////////////////////////////////////////////////////////
    //acf-bsca initialization
    int iter = 1;
    int* index= new int[2*number_of_training_points];
    unsigned long long steps = 0;
    int max_iter= 15000;
    double eps=0.0001;
    int slen ;
    double dualVartemp=0.0;
    double prefsum=0.0;
    // prepare preferences for scheduling
    double* pref = new double[number_of_training_points]();
    const double gain_learning_rate = 1.0 / (number_of_training_points);
    double average_gain = 0.0;
    //bool canstop = true;
    double stopping = INFINITY;
    //////////////////////////////////////////////////////////
    //acf-bsca prepare data
    for (size_t i=0; i<number_of_training_points; i++) pref[i] = rand()/number_of_training_points ; //rand() % 20;
    for (size_t i=0; i<number_of_training_points; i++) prefsum+=pref[i];
    double nOversum = number_of_training_points / prefsum;
    double* acc = new double[number_of_training_points]() ;
    //std::fill_n(acc, number_of_training_points, 0.0);
    //std::fill_n(pref, number_of_training_points, 0.0);
    for (size_t i=0; i<number_of_training_points; i++)
    {
        double a    = acc[i] + nOversum * pref[i];
        int n       = (int)floor(a);
        acc[i]      = a - n;
    }
    
    //////////////////////////////////////////////////////////
    INDEX numIter = 1;
    int num_iter = 1;
    size_t epoch = 0;
    double storage_old_pseudoGmin = 0.0;
    double storage_new_pseudoGmin = 0.0;
    double storage_old_pseudoGmax = 0.0;
    double storage_new_pseudoGmax = 0.0;
    //#while (std::fabs(stopping) > 0.00001)
    for (size_t epoch = 0; epoch < max_epochs; epoch++)
    {
        //dataset.shuffle_ds_dualvec(dual_variables);
        vector<double> pseudo_variables_size(number_of_training_points, 0.0);
        int variable = 0;
        double storage_old_pseudoGmin ;
        double storage_new_pseudoGmin ;
        double storage_old_pseudoGmax ;
        double storage_new_pseudoGmax ;
        cout << endl << epoch+1 << ":";
        double gradientMIN = INFINITY;
        double gradientMAX = -INFINITY;
        
        double KL_max = -INFINITY;
        sequence.clear();
        fillUniformSequence(sequence, number_of_training_points);
        //size_t sequence_size = sequence.size();
        //double maxPseudoinsequence = -INFINITY;
        //double gradienttime_counter= 0.0;
        double mergeAndDeleteSV_counter = 0.0;
        //double gainbasedpref_time_counter = 0.0;
        double scheduling_time_counter=0.0;
        unsigned int countMerges = 0;
        
        //start training:
        train_start_t = (double)clock() / CLOCKS_PER_SEC;
        double maxPseudo = 0;
        
        unsigned int counter = 0;
        vector<double> extradualvariables;
        sparseData extradataset;
        
        /////////////////////////////////////////////////////////////////////
        //Scheduling start
        double scheduling_start_t = (double)clock() / CLOCKS_PER_SEC;
        slen = 0.0;
        nOversum = number_of_training_points / prefsum;
        
        for (INDEX i = 0; i < number_of_training_points; i++)
        {
            numIter++;
            double a    = acc[i] + nOversum * pref[i];
            int n       = (int)floor(a);
            for (int j=0; j<n; j++){index[slen] = i; slen++;}
            acc[i]      = a - (double)n;
            //cout << acc[i] << endl;
        }
        double scheduling_end_t = (double)clock() / CLOCKS_PER_SEC;
        scheduling_time_counter+=scheduling_end_t-scheduling_start_t;
        // double scheduling_start_t = (double)clock() / CLOCKS_PER_SEC;
        // define schedule
        //std::tie(slen, acc, index, ws_sequence) = scheduling(number_of_training_points, slen, prefsum, acc, pref, index);
        //double scheduling_end_t = (double)clock() / CLOCKS_PER_SEC;
        //scheduling_time_counter+=scheduling_end_t-scheduling_start_t;
        //Scheduling end
        /////////////////////////////////////////////////////////////////////
        //for (int s=0; s<slen;s++ )ws_sequence.push_back(index[s]); //the size of ws will be changing all the time
        ws_sequence.clear();
        for (int s=0; s<slen;s++ )
        {
            ws_sequence.push_back(index[s]);
        }
        for (unsigned int i=0; i<ws_sequence.size(); i++)
        {
            //swap(ws_vector[i], ws_vector[rand() % ws_vector.size()]);
            swap(ws_sequence[i], ws_sequence[rand() % ws_sequence.size()]);
        }
        
        cout << "ws size is: " << ws_sequence.size();
        size_t ws_sequence_size = ws_sequence.size();
        steps +=ws_sequence_size;
        
        dfile_wsSize << "epoch:" << epoch+1 << ":wsSize:" << ws_sequence_size << endl;
        
        //fillUniformSequence(ws_sequence, ws_sequence_size);
        
        for (INDEX i = 0; i < ws_sequence_size; i++)
        {
            INDEX ws = ws_sequence[i];
            double alpha_point = dual_variables[ws];
            //double gradient = computeGradient(dataset.data[ws], dataset.labels[ws], pseudo_variables, pseudo, kernel);
            //double G = 0;
            vector<double> gradientminmax;
            gradientminmax = computeGradientPseudoMaxMin(alpha_point, dataset.data[ws], dataset.labels[ws], pseudo_variables, pseudo, kernel);
            
            double gradient = gradientminmax[0];
            double gradientmax = gradientminmax[1];
            double gradientmin = gradientminmax[2];
            
            if (gradientmin < gradientMIN) gradientMIN = gradientmin;
            if (gradientmax > gradientMAX) gradientMAX = gradientmax;
            ///////////////////////////////
            //vector<double>p_matrix;
            //p_matrix = computeS_studentDistribution(dataset.data[ws], dataset.labels[ws], pseudo_variables, pseudo, kernel);
            //INDEX gradient_inx = p_matrix.size()-1;
            //double gradient = p_matrix[gradient_inx];
            //cout << gradient << "; " << gradient1 << endl;
            //G = -gradient;
            //double gain_g = -gradient;
            double gain = 0.0;
            double change = 0.0;
            double newpref;
            ////////////////////////////////
            
            //cout << maxPseudo << endl;
            double old_alpha = dual_variables[ws];
            double new_alpha = max(0.0, min(old_alpha + gradient, C));
            dual_variables[ws] = new_alpha;
            
            if(new_alpha == 0) dualvarisZero+=1;
            if(new_alpha == C) dualvarisC+=1;
            if(new_alpha<C && new_alpha>0) dualvarbetZeroC+=1;
            
            //////////////////////////////////////////////////////////////////////
            // dual variable changes
            if (new_alpha != old_alpha)
            {
                pseudo.data.push_back(dataset.data[ws]);
                pseudo_variables.push_back(new_alpha-old_alpha);
                pseudo.labels.push_back(dataset.labels[ws]);
                double delta = new_alpha - old_alpha;
                gain = delta * (gradient - 0.5 * delta);
            }
            //////////////////////////////////////////////////////////////////////
            //cout << "\n" << numIter;
            if (numIter == 0) average_gain += gain/(double)slen;
            else{
                change = CHANGE_RATE * (gain/average_gain - 1.0);
                newpref = min(PREF_MAX, max(PREF_MIN, pref[ws] * exp(change)));
                prefsum += newpref - pref[ws];
                pref[ws] = newpref;
                average_gain = (1.0 - gain_learning_rate) * average_gain + gain_learning_rate * gain;
            }
            //////////////////////////////////////////////////////////////////////
            while (pseudo.data.size() > B)
            {
                counter++;
                
                //Check the model size compared to the budget
                double mergeAndDeleteSV_start_time = (double)clock() / CLOCKS_PER_SEC;
                mergeAndDeleteSV(pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);
                double mergeAndDeleteSV_end_time = (double)clock() / CLOCKS_PER_SEC;
                mergeAndDeleteSV_counter += mergeAndDeleteSV_end_time - mergeAndDeleteSV_start_time;
                countMerges++;
                
            }
            
            numIter++;
            
            
        } //end of sequence
        //epoch++;
        num_iter++;
        double train_end_t = (double)clock() / CLOCKS_PER_SEC;
        //dobjfn_dfile_merging << "epoch :"  << epoch +1 << ":KL:" << KL_max;
        dobjfn_dfile_merging << "epoch :"  << epoch +1  <<  ":mergingtime:" << mergeAndDeleteSV_counter << ":mergingsteps:" << countMerges << endl;
        
        dobjfn_dfile_gradvariants << "epoch :"  << epoch +1  << ":maxG_pseudovar:"  << maxG_pseudovar<< ":avg gradient:" << maxG_pseudovar/sum_maxG_pseudovar << ":minPgradient:" << gradientMIN << ":maxPgradient:" << gradientMAX <<  ":gradientpseudoMIN:" << gradientpseudoMIN << ":gradientpseudoMAX:" << gradientpseudoMAX  << endl;
        countMerges = 0;
        mergeAndDeleteSV_counter = 0.0;
        
        SVM svm(pseudo_variables, pseudo, kernel);
        dobjfn_dfile_per << "epoch :"  << epoch +1  << ": per.:"  << svm.evaluateTestset(testdataset)<< ":" << endl;
        
        unsigned int nSV = 0; unsigned int nBSV = 0;
        for (size_t i=0; i<number_of_training_points; i++)
        {
            if(fabs(dual_variables[i])>0)
            {
                ++nSV;
                if(dataset.labels[i]>0)
                {
                    if(fabs(dual_variables[i])>=C)++nBSV;
                    
                } else
                {
                    if(fabs(dual_variables[i])>=0)++nBSV;
                }
            }
            
        }
        dobjfn_dfile_dualpseudoCounter <<"epoch :"  << epoch +1 << ":nSV: "<<nSV<<":nBSV:"<<nBSV<< endl;
        
        dobjfn_dfile_traint << "epoch :"  << epoch +1  << ":trainingtime.:"  << train_end_t - train_start_t << ":" << endl;
        //dobjfn_dfile_maxG << "epoch :"  << epoch +1  << ":maxG:" << gradientMAX << ":minG:" << gradientMIN << endl;  <<  ":dualvarisZero.:"  << dualvarisZero << ":dualvarisC.:"  << dualvarisC << ":dualvarbetZeroC.:"  << dualvarbetZeroC<< ":" << endl;
        dualvarisZero  = 0;
        dualvarisC     = 0;
        dualvarbetZeroC = 0;
        
        /* dualObjFunValue =  \sum dual-variables - 0.5 (minW)^2 */
        std::tie(dualObjFunValue, dualVariable, dual_05_minWsquare) = dualObjectiveFunction ( C, pseudo_variables, dual_variables, pseudo, dataset, kernel );
        dobjfn_dfile  << "epoch:"<< epoch << ":" <<  dualObjFunValue << endl;
        //dobjfn_dfile_param << "epoch:"  <<epoch +1  << ":sumOfalpha:" << dualVariable << ":(1/2)minWsquare:" << dual_05_minWsquare << ":"  << endl;
        /* primalObjFunValue =  C*HLoss + 0.5 (minW)^2 */
        std::tie(primalObjFunValue, primal_05_minWsquare, primal_C_mul_Hloss) = primalObjectiveFunction (C, pseudo_variables, dual_variables, pseudo, dataset, kernel );
        pobjfn_dfile  << "epoch:"<< epoch << ":" <<  primalObjFunValue << endl;
        //pobjfn_dfile_param << "epoch:"  <<epoch +1  << ":C*HLoss:" << primal_C_mul_Hloss << ":(1/2)minWsquare:" << primal_05_minWsquare << ":" << endl;
        double old_diff = 0.0;
        if (num_iter ==2)
        {
            storage_new_pseudoGmax = gradientMAX;
            storage_new_pseudoGmin = gradientMIN;
        }
        else {
            storage_old_pseudoGmax = storage_new_pseudoGmax;
            storage_old_pseudoGmin = storage_new_pseudoGmin;
            storage_new_pseudoGmax = gradientMAX;
            storage_new_pseudoGmin = gradientMIN;
            old_diff = storage_old_pseudoGmax - storage_old_pseudoGmin;
        }
        double new_diff = storage_new_pseudoGmax - storage_new_pseudoGmin;
        //old_diff = storage_old_pseudoGmax - storage_old_pseudoGmin;
        double new_diff_log = std::log(new_diff);
        double old_diff_log;
        if (num_iter ==2)
        {
            old_diff_log = 0.0;
        }
        else
            old_diff_log = std::log(old_diff);
        stopping = new_diff_log - old_diff_log;
        ws_sequence.clear() ;
        dobjfn_dfile_maxG << "epoch :"  << epoch +1  << ":maxG:" << gradientMAX << ":minG:" << gradientMIN << ":stopping:" << stopping << endl;
        //if(old_diff_log == INFINITY) continue;
    }
    dobjfn_dfile.close();
    pobjfn_dfile.close();
    
    dobjfn_dfile_param.close();
    pobjfn_dfile_param.close();
    
    dobjfn_dfile_per.close();
    dobjfn_dfile_traint.close();
    dobjfn_dfile_merging.close();
    
    dfile_wsSize.close();
    dobjfn_dfile_maxwd.close();
    dobjfn_dfile_maxG.close();
    dobjfn_dfile_minG.close();
    dobjfn_dfile_maxpseudoG.close();
    dobjfn_dfile_minpseudoG.close();
    
    return SVM(pseudo_variables, pseudo, kernel);
}

//Budgeted dual solver
SVM SBSCA(sparseData& dataset, sparseData& testdataset, double C, Kernel& kernel, LookupTable& wd_parameters, double accuracy, size_t B, size_t max_epochs, Heuristic heuristic)
{
    cout << "Optimisation ... begin \n";
    size_t number_of_training_points = dataset.data.size();
    cout << "Number of training points: " << number_of_training_points << endl;
    // unsigned int iterFreq = 100000;
    sparseData pseudo;
    vector<double> pseudo_variables;
    
    vector<INDEX> sequence(0);
    vector<double> dual_variables(number_of_training_points, 0);
    std::string budget = to_string(B);
    std::string datasetname;
#ifdef adult
    datasetname = "a";
    cout << "dataset: adult\n";
#endif
#ifdef codrna
    datasetname = "c";
    cout << "dataset: codrna\n";
#endif
#ifdef susy
    datasetname = "s";
    cout << "dataset: susy\n";
#endif
#ifdef covtype
    datasetname = "cov";
    cout << "dataset: covtype\n";
#endif
    
    
    // Main optimization loop
    //double dualObjFunValue = 0.0 , primalObjFunValue = 0.0;
    //double dualVariable = 0.0, dual_05_minWsquare = 0.0;
    //double primal_05_minWsquare = 0.0 , primal_C_mul_Hloss = 0.0;
    std::string PATH;
#ifdef FREEZE
    PATH = "freeze2/";
    //system("mkdir \"a_b500_freeze\"");
    system("mkdir \"freeze2\"") ;
    cout << "Freeze mode\n";
#endif
#ifdef PROJECTION
    PATH = "projection2/";
    system("mkdir \"projection2\"");
    cout << "Projection mode\n";
#endif
#ifdef MERGE
    PATH = "svmguideP20/";
    system("mkdir \"svmguideP20\"");
    cout << "Merging mode\n";
#endif
    
    cout << "Max Epochs: " << max_epochs << "  Current:";
    ofstream dobjfn_dfile;
    dobjfn_dfile.open (PATH + "dualobjective.txt");
    //dobjfn_dfile << "Writing this to a file.\n";
    ofstream dobjfn_dfile_param;
    dobjfn_dfile_param.open (PATH + "dualparameters.txt");
    //dobjfn_dfile_param << "Writing this to a file.\n";
    ofstream dobjfn_dfile_per;
    dobjfn_dfile_per.open (PATH + "testaccuracy.txt");
    
    //dobjfn_dfile_per << "Writing this to a file.\n";
    ofstream dobjfn_dfile_pseudoVariables;
    dobjfn_dfile_pseudoVariables.open (PATH + "pseudovariables.txt");
    //dobjfn_dfile_pseudoVariables << "Writing this to a file.\n";
    ofstream dobjfn_dfile_dualVariables;
    dobjfn_dfile_dualVariables.open (PATH + "dualvariables.txt");
    
    ofstream dobjfn_dfile_gradStepdual;
    dobjfn_dfile_gradStepdual.open (PATH + "rowstoppingparameters.txt");
    //dobjfn_dfile_pseudoLabels << "Writing this to a file.\n";
    ofstream dobjfn_dfile_gradSteppseudo;
    dobjfn_dfile_gradSteppseudo.open (PATH + "pseudostoppingparameters.txt");
    //dobjfn_dfile_pseudoLabels << "Writing this to a file.\n";
    ofstream dobjfn_dfile_dualpseudoCounter;
    dobjfn_dfile_dualpseudoCounter.open (PATH + "supportvectorscounter.txt");
    
    
    ofstream pobjfn_dfile;
    pobjfn_dfile.open (PATH + "primalobjective.txt");
    //pobjfn_dfile << "Writing this to a file.\n";
    ofstream pobjfn_dfile_param;
    pobjfn_dfile_param.open (PATH + "primalparameters.txt");
    //pobjfn_dfile_param << "Writing this to a file.\n";
    
    ofstream dobjfn_dfile_traint;
    dobjfn_dfile_traint.open (PATH + "trainingtime.txt");
    //dobjfn_dfile_traint << "Writing this to a file.\n";
    
    ofstream dobjfn_dfile_merging;
    dobjfn_dfile_merging.open (PATH + "merging.txt");
    //dobjfn_dfile_merging << "Writing this to a file.\n";
    
    double train_start_t = 0.0;
    //double train_end_t = 0.0;
    // bool BreakConditionON = false;
    
    double mergeAndDeleteSV_counter = 0.0;
    unsigned int countMerges = 0;
    
    unsigned int dualvarisZero = 0;
    unsigned int dualvarisC = 0;
    unsigned int dualvarbetZeroC = 0;
    
    INDEX numIter = 1;
    size_t M = 20;                            //200//500 //%4800;           % (EigenPro) subsample size.
    unsigned int topk = 16;                   //160 //10//%160;            % (EigenPro) top-k eigensystem.
    size_t n = number_of_training_points;       //size(train_x, 1); % number of training samples.
    // size_t N = 24;                             //size(train_x, 2); % number of raw features.
    
    for (size_t epoch = 0; epoch < max_epochs; epoch++)
    {
        //dataset.shuffle_ds_dualvec(dual_variables);
        if(epoch == 5){// || epoch == 10){// || epoch == 500 || epoch == 1000 || epoch == 1500 || epoch == 2000 || epoch == 2500 ||epoch == 3000 || epoch ==3500 || epoch == 4000 || epoch == 4500){
            pseudo_variables.clear();
            pseudo.data.clear();
            pseudo.labels.clear();
        }
        cout << endl << epoch+1 << ":";
        double gradientMIN = INFINITY;
        double gradientMAX = -INFINITY;
        double mMIN = INFINITY;
        double mMAX = -INFINITY;
        double pseudo_mMAX = -INFINITY;
        INDEX pseudoGMAX;
        double KL_max = -INFINITY;
        sequence.clear();
        //fillUniformSequence(sequence, number_of_training_points);
        
        double maxPseudoinsequence = -INFINITY;
        train_start_t = (double)clock() / CLOCKS_PER_SEC;
        double maxPseudo = 0.0;
        double maxG_pseudovar = -INFINITY;
        double sum_maxG_pseudovar = 0.0;
        
        unsigned int counter = 0;
        vector<double> extradualvariables;
        sparseData extradataset;
        sequence.clear();
        for (INDEX i=0; i<number_of_training_points; i++)
        {
            sequence.push_back(i);
        }
        for (unsigned int i=0; i<number_of_training_points; i++)
        {
            //swap(ws_vector[i], ws_vector[rand() % ws_vector.size()]);
            swap(sequence[i], sequence[rand() % number_of_training_points]);
        }
        size_t sequence_size = sequence.size();
        int lr = 1;
        
        
        
        std::vector<double> s;
        //MatrixXf V;
        double lambda1;
        double s_sum = 0.0;
        
        double eigenpro_eta = 0.0;
        double conditionNumber ;
        
        for (INDEX i = 0; i < sequence_size; i++)
        {
            numIter++;
            
            INDEX ws = sequence[i];
            double alpha_point = dual_variables[ws];
            /*
             //Compute gradient
             //vector<double> gradient_maxPseudo;
             //gradient_maxPseudo =computeGradient(dataset.data[ws], dataset.labels[ws], pseudo_variables, pseudo, kernel);
             */
            //double gradient = computeGradient(dataset.data[ws], dataset.labels[ws], pseudo_variables, pseudo, kernel);
            vector<double> gradientminmax;
            gradientminmax = computeGradientPseudoMaxMin(alpha_point, dataset.data[ws], dataset.labels[ws], pseudo_variables, pseudo, kernel);
            
            double gradient = gradientminmax[0];
            double gradientmax = gradientminmax[1];
            double gradientmin = gradientminmax[2];
            double pseudomax   = gradientminmax[3];
            //double pseudomin   = gradientminmax[4];
            INDEX pseudomaxinx = (int) pseudomax;
            
            double m_max   = gradientminmax[5];
            double m_min   = gradientminmax[6];
            //pseudo_variables[pseudomaxinx].clear();
            
            double pseudo_m = gradientminmax[7];
            
            if (gradientmin < gradientMIN) gradientMIN = gradientmin;
            if (gradientmax > gradientMAX) {gradientMAX = gradientmax; pseudoGMAX = pseudomaxinx;}
            if (m_min < mMIN) mMIN = m_min;
            if (m_max > mMAX) {mMAX = m_max; }
            
            if (pseudo_m > pseudo_mMAX) {pseudo_mMAX = pseudo_m; }
            
            double old_alpha = dual_variables[ws];
            double new_alpha = max(0.0, min(old_alpha + gradient, C));
            dual_variables[ws] = new_alpha;
            
            
            
            ///////////////////////////////
            vector<double>p_matrix;
            
            
            
            
            if(new_alpha == 0) dualvarisZero+=1;
            if(new_alpha == C) dualvarisC+=1;
            if(new_alpha<C && new_alpha>0) dualvarbetZeroC+=1;
            //////////////////////////////////////////////////////////////////////
            double absG_pseudovar = 0.0;
            int score = 0;
            if (old_alpha == 0.0 && gradient <= 0.0){}
            else if (old_alpha == C && gradient >= 0.0){}
            else
            {
                absG_pseudovar = std::abs(gradient);
                (absG_pseudovar > maxG_pseudovar)? score=1 : score=2 ;
                if (absG_pseudovar > maxG_pseudovar) maxG_pseudovar = absG_pseudovar;
                //sumG_dualvar computation
                sum_maxG_pseudovar += absG_pseudovar;
                
            }
            
            
            
            if ( (new_alpha != old_alpha) )
                //if ((new_alpha != old_alpha) && epoch <= 1 )
            {
                pseudo.data.push_back(dataset.data[ws]);
                pseudo_variables.push_back((new_alpha-old_alpha)); //-old_alpha
                pseudo.labels.push_back(dataset.labels[ws]);
                
            }
            
            
            //////////////////////////////////////////////////////////////////////
            
            if (pseudo.data.size() > B  && epoch <= 1)
            {
                counter++;
                
                //Check the model size compared to the budget
                double mergeAndDeleteSV_start_time = (double)clock() / CLOCKS_PER_SEC;
               // mergeAndDeleteSV(pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);
                projectSV(pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);
                double mergeAndDeleteSV_end_time = (double)clock() / CLOCKS_PER_SEC;
                mergeAndDeleteSV_counter += mergeAndDeleteSV_end_time - mergeAndDeleteSV_start_time;
                countMerges++;
                
            }
            
            else if (pseudo.data.size() > B && epoch >1  )
            {
                
                if(score == 1 || score == 2)
                    
                {
                    
                    //#ifdef MERGE
                    
                    counter++;
                    //pseudo_variables.pop_back();
                    //pseudo_variables.push_back((new_alpha-old_alpha)*lambda1); //-old_alpha
                    //Check the model size compared to the budget
                    double mergeAndDeleteSV_start_time = (double)clock() / CLOCKS_PER_SEC;
                    mergeAndDeleteSV(pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);
                    double mergeAndDeleteSV_end_time = (double)clock() / CLOCKS_PER_SEC;
                    mergeAndDeleteSV_counter += mergeAndDeleteSV_end_time - mergeAndDeleteSV_start_time;
                    countMerges++;
                    
                }
                
                
                else if(score ==3)
                {
                    
                    counter++;
                    //  pseudo_variables.pop_back();
                    //pseudo_variables.push_back((new_alpha-old_alpha)*lambda1);
                    //Check the model size compared to the budget
                    double mergeAndDeleteSV_start_time = (double)clock() / CLOCKS_PER_SEC;
                    //cout << "pseudo_variables: " << pseudo_variables[B];
                    projectSV(pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);
                    
                    double mergeAndDeleteSV_end_time = (double)clock() / CLOCKS_PER_SEC;
                    mergeAndDeleteSV_counter += mergeAndDeleteSV_end_time - mergeAndDeleteSV_start_time;
                    countMerges++;
                    
                    // }
                    //#endif
                    
                }
            }
            score = 0;
            
            
        } //end of sequence
        
       /* std::tie(s, V, lambda1)=  rsvd ( pseudo, kernel, M,  topk);
        eigenpro_eta = 1.5 / (2 * s[0] / n);
        conditionNumber = s[0]/s[10];
        dobjfn_dfile_pseudoVariables <<"epoch :"  << epoch +1  << ":eigenpro_eta:"  << eigenpro_eta<< ":conditionNumber:" << conditionNumber << ":lambda1:"<< lambda1 << endl;
        */
        
        double train_end_t = (double)clock() / CLOCKS_PER_SEC;
        //dobjfn_dfile_merging << "epoch :"  << epoch +1 << ":KL:" << KL_max;
        
        dobjfn_dfile_merging << "epoch :"  << epoch +1  << ":maxG_pseudovar:"  << maxG_pseudovar<< ":avg gradient:" << maxG_pseudovar/sum_maxG_pseudovar << ":minPgradient:" << gradientMIN << ":maxPgradient:" << gradientMAX << ":m_min:" << mMIN << ":m_max:" << mMAX  << ":pseudo_mMAX:" << pseudo_mMAX <<  ":mergingtime:" << mergeAndDeleteSV_counter << ":mergingsteps:" << countMerges << endl;
        
        
        countMerges = 0;
        sum_maxG_pseudovar = 0.0;
        maxG_pseudovar = -INFINITY;
        
        SVM svm(pseudo_variables, pseudo, kernel);
        dobjfn_dfile_per << "epoch :"  << epoch +1  << ": per.:"  << svm.evaluateTestset(testdataset)<< ":" << endl;
        unsigned int nSV = 0; unsigned int nBSV = 0;
        for (size_t i=0; i<number_of_training_points; i++)
        {
            if(fabs(dual_variables[i])>0)
            {
                ++nSV;
                if(dataset.labels[i]>0)
                {
                    if(fabs(dual_variables[i])>=C)++nBSV;
                    
                } else
                {
                    if(fabs(dual_variables[i])>=0)++nBSV;
                }
            }
            
        }
        dobjfn_dfile_dualpseudoCounter <<"epoch :"  << epoch +1 << ":nSV: "<<nSV<<":nBSV:"<<nBSV<< endl;
        
        
        
        dobjfn_dfile_traint << "epoch :"  << epoch +1  << ":trainingtime.:"  << train_end_t - train_start_t << ":" << endl;
        
        
        /* dualObjFunValue =  \sum dual-variables - 0.5 (minW)^2 */
        double dualObjFunValue = 0.0;
        double dualVariable = 0.0;
        double dual_05_minWsquare = 0.0;
        std::tie(dualObjFunValue, dualVariable, dual_05_minWsquare) = dualObjectiveFunction ( C, pseudo_variables, dual_variables, pseudo, dataset, kernel );
        dobjfn_dfile <<"epoch:"  << epoch +1 << ":" << dualObjFunValue << ":" << endl;
        dobjfn_dfile_param << "epoch:"  <<epoch +1  << ":sumOfalpha:" << dualVariable << ":(1/2)minWsquare:" << dual_05_minWsquare << ":"  << endl;
        dualObjFunValue = 0.0;
        dualVariable = 0.0;
        dual_05_minWsquare = 0.0;
        /* primalObjFunValue =  C*HLoss + 0.5 (minW)^2 */
        double primalObjFunValue = 0.0;
        double primal_05_minWsquare = 0.0;
        double primal_C_mul_Hloss = 0.0;
        
        std::tie(primalObjFunValue, primal_05_minWsquare, primal_C_mul_Hloss) = primalObjectiveFunction (C, pseudo_variables, dual_variables, pseudo, dataset, kernel );
        pobjfn_dfile <<"epoch:"  <<epoch +1 << ":" << primalObjFunValue << ":" << endl;
        pobjfn_dfile_param << "epoch:"  <<epoch +1  << ":C*HLoss:" << primal_C_mul_Hloss << ":(1/2)minWsquare:" << primal_05_minWsquare << ":" << endl ;
        
        primalObjFunValue = 0.0;
        primal_05_minWsquare = 0.0;
        primal_C_mul_Hloss = 0.0;
        
        dualvarisZero  = 0;
        dualvarisC     = 0;
        dualvarbetZeroC = 0;
        
        
    }
    dobjfn_dfile.close();
    pobjfn_dfile.close();
    
    dobjfn_dfile_param.close();
    pobjfn_dfile_param.close();
    
    dobjfn_dfile_per.close();
    dobjfn_dfile_traint.close();
    dobjfn_dfile_merging.close();
    
    dobjfn_dfile_pseudoVariables.close();
    
    
    return SVM(pseudo_variables, pseudo, kernel);
}


SVM BMVPSMOSimplified(sparseData& dataset, sparseData& testdataset, double C, Kernel& kernel, LookupTable& wd_parameters, double accuracy, size_t B, size_t max_epochs, Heuristic heuristic)
{
    cout << "Optimisation ... begin \n";
    size_t number_of_training_points = dataset.data.size();
    cout << "Number of training points: " << number_of_training_points << endl;
    // unsigned int iterFreq = 100000;
    sparseData pseudo;
    vector<double> pseudo_variables;
    
    vector<INDEX> sequence(0);
    vector<double> dual_variables(number_of_training_points, 0);
    
    // Main optimization loop
    //double dualObjFunValue = 0.0 , primalObjFunValue = 0.0;
    //double dualVariable = 0.0, dual_05_minWsquare = 0.0;
    //double primal_05_minWsquare = 0.0 , primal_C_mul_Hloss = 0.0;
    
    std::string PATH = "Exp1_SMOs_B50/";
    system("mkdir \"Exp1_SMOs_B50\"");
    cout << "Max Epochs: " << max_epochs << "  Current:";
    ofstream dobjfn_dfile;
    dobjfn_dfile.open (PATH + "dualobjective.txt");
    //dobjfn_dfile << "Writing this to a file.\n";
    ofstream dobjfn_dfile_param;
    dobjfn_dfile_param.open (PATH + "dualparameters.txt");
    //dobjfn_dfile_param << "Writing this to a file.\n";
    ofstream dobjfn_dfile_per;
    dobjfn_dfile_per.open (PATH + "testaccuracy.txt");//dualparameters.txt testaccuracy.txt pseudovariables.txt dualvariables.txt rowstoppingparameters.txt pseudostoppingparameters.txt supportvectorscounter.txt primalobjective.txt primalparameters.txt trainingtime.txt merging.txt
    
    //dobjfn_dfile_per << "Writing this to a file.\n";
    ofstream dobjfn_dfile_pseudoVariables;
    dobjfn_dfile_pseudoVariables.open (PATH + "pseudovariables.txt");
    //dobjfn_dfile_pseudoVariables << "Writing this to a file.\n";
    ofstream dobjfn_dfile_dualVariables;
    dobjfn_dfile_dualVariables.open (PATH + "dualvariables.txt");
    //dobjfn_dfile_dualVariables << "Writing this to a file.\n";
    //ofstream dobjfn_dfile_pseudoData;
    //dobjfn_dfile_pseudoData.open (PATH + "dpseudoData_dsolver.txt");
    //dobjfn_dfile_pseudoData << "Writing this to a file.\n";
    //ofstream dobjfn_dfile_pseudoLabels;
    //dobjfn_dfile_pseudoLabels.open (PATH + "dpseudoLabels_dsolver.txt");
    //dobjfn_dfile_pseudoLabels << "Writing this to a file.\n";
    ofstream dobjfn_dfile_gradStepdual;
    dobjfn_dfile_gradStepdual.open (PATH + "rowstoppingparameters.txt");
    //dobjfn_dfile_pseudoLabels << "Writing this to a file.\n";
    ofstream dobjfn_dfile_gradSteppseudo;
    dobjfn_dfile_gradSteppseudo.open (PATH + "pseudostoppingparameters.txt");
    //dobjfn_dfile_pseudoLabels << "Writing this to a file.\n";
    ofstream dobjfn_dfile_dualpseudoCounter;
    dobjfn_dfile_dualpseudoCounter.open (PATH + "supportvectorscounter.txt");
    
    
    ofstream pobjfn_dfile;
    pobjfn_dfile.open (PATH + "primalobjective.txt");
    //pobjfn_dfile << "Writing this to a file.\n";
    ofstream pobjfn_dfile_param;
    pobjfn_dfile_param.open (PATH + "primalparameters.txt");
    //pobjfn_dfile_param << "Writing this to a file.\n";
    
    ofstream dobjfn_dfile_traint;
    dobjfn_dfile_traint.open (PATH + "trainingtime.txt");
    //dobjfn_dfile_traint << "Writing this to a file.\n";
    
    ofstream dobjfn_dfile_merging;
    dobjfn_dfile_merging.open (PATH + "merging.txt");
    //dobjfn_dfile_merging << "Writing this to a file.\n";
    
    double train_start_t = 0.0;
    //double train_end_t = 0.0;
    // bool BreakConditionON = false;
    
    double mergeAndDeleteSV_counter = 0.0;
    unsigned int countMerges = 0;
    //double sum_diffsteps_dualvar = 0.0;
    //double sum_diffsteps_pseudovar = 0.0;
    //double sum_maxG_dualvar = 0.0;
    //double sum_maxG_pseudovar = 0.0;
    
    //double maxG_dualvar = -INFINITY;
    //double maxG_pseudovar = -INFINITY;
    //double maxStep_dualvar = -INFINITY;
    //double maxStep_pseudovar = -INFINITY;
    
    unsigned int dualvarisZero = 0;
    unsigned int dualvarisC = 0;
    unsigned int dualvarbetZeroC = 0;
    
    
    //INDEX ws;
    
    //size_t number_of_support_vectors = 0;
    vector<double> gradients(number_of_training_points, 1.0);
    
    
    //size_t iteration_counter = 0;
    // Initialize constraints for y*alpha
    vector<double> lower_constraints_combined(number_of_training_points, 0);
    vector<double> upper_constraints_combined(number_of_training_points, 0);
    for (INDEX i = 0; i < number_of_training_points; i++) {
        if (dataset.labels[i] == 1) {
            lower_constraints_combined[i] = 0;
            upper_constraints_combined[i] = C;
        } else if (dataset.labels[i] == -1) {
            lower_constraints_combined[i] = -C;
            upper_constraints_combined[i] = 0;
        }
    }
    
    vector<double>ygi_storage;
    vector<double>ygj_storage;
    //int counterij = 0;
    for(int ii=0;ii<max_epochs; ii++)
        // while (true)
    {
        tuple<INDEX, INDEX, bool> working_set = findMVPs(dataset.labels, gradients, dual_variables, lower_constraints_combined, upper_constraints_combined, accuracy);
        INDEX i, j, ws;
        i = get<0>(working_set);
        j = get<1>(working_set);
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
        //Optimality criterion (relaxed) (instead of yg_i <= yg_j we use yg_i - yg_j < epsilon like LIBSVM)
        //eps is 0.001 in LIBSVM, cf. http://www.csie.ntu.edu.tw/~r94100/libsvm-2.8/README
        
        
        ygi_storage.push_back(yg_i);
        ygj_storage.push_back(yg_j);
        cout << std::abs(yg_i) << ":"<<std::abs(yg_j)<<endl;
        if (gradients[i] < gradients[j])
            //if (abs(yg_j) < abs(yg_i))
        {
            ws = i;
        }
        else {
            ws = j;
        }
        
        
        // Compute gradient!
        double gradient = computeGradient(dataset.data[ws], dataset.labels[ws], pseudo_variables, pseudo, kernel);
        
        // Optimize dual objective function over the chosen working set (direction search / newton step)
        double newton_max = gradient;
        
        // Truncate optimum (dual objective function is concave, so naive truncation makes sense)
        double old_alpha = dual_variables[ws];
        double new_alpha = max(0.0, min(old_alpha + newton_max, C));
        
        /*
         counterij++;
         if(counterij > 4 && pseudo.data.size() > (B-1) )
         {
         
         size_t ygi_storage_size = ygi_storage.size()-1;
         size_t ygj_storage_size = ygj_storage.size()-1;
         if(ygi_storage[ygi_storage_size] + ygi_storage[ygi_storage_size-1] - 2*ygi_storage[ygi_storage_size-2]<0.0000001)
         {
         if(ygj_storage[ygj_storage_size] + ygj_storage[ygj_storage_size-1] - 2*ygj_storage[ygj_storage_size-2]<0.0000001)
         {
         cout << "now it should break\n";
         break;
         }
         }
         }
         */
        //Direction Search
        // double newton_min = (yg_i - yg_j)/(kernel.evaluate(dataset.data[i], dataset.data[i]) + kernel.evaluate( dataset.data[j], dataset.data[j]) - 2*kernel.evaluate(dataset.data[i],  dataset.data[j]));
        // double lambda = min(B_i - ya_i, min(ya_j - A_j, newton_min));
        // Gradient Update
        // for (INDEX index = 0; index < gradients.size(); index++) {
        //   double gradient_change = lambda*dataset.labels[index]*(kernel.evaluate(dataset.data[j], dataset.data[index]) - kernel.evaluate(dataset.data[i], dataset.data[index]));
        //  gradients[index] += gradient_change;
        // }
        
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
        
        
        //Dual variable update
        dual_variables[ws] = new_alpha;
        
        // Keep track of SVs
        // if ((old_alpha == 0) && (new_alpha != 0)) {
        if ((old_alpha - new_alpha != 0)) {
            
            
            // Maintain Pseudorepresentation
            pseudo.data.push_back(dataset.data[ws]);
            pseudo_variables.push_back(new_alpha - old_alpha );
            pseudo.labels.push_back(dataset.labels[ws]);
            
            // pseudo.data.push_back(dataset.data[j]);
            // pseudo_variables.push_back(new_alpha_j - old_alpha_j );
            // pseudo.labels.push_back(dataset.labels[j]);
            
        }
        
        bool gradient_change = false;
        while (pseudo.data.size() > B)
        {
            
            //Check the model size compared to the budget
            double mergeAndDeleteSV_start_time = (double)clock() / CLOCKS_PER_SEC;
            mergeAndDeleteSV(pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);
            //DeleteSV(pseudo_variables,  pseudo, kernel, wd_parameters, C, heuristic);
            
            
            double mergeAndDeleteSV_end_time = (double)clock() / CLOCKS_PER_SEC;
            mergeAndDeleteSV_counter += mergeAndDeleteSV_end_time - mergeAndDeleteSV_start_time;
            countMerges++;
            gradient_change = true;
            
        }
        
        for (INDEX index = 0; index < gradients.size(); index++)
        {
            //gradients[index] -= dataset.labels[ws]*dataset.labels[index]*kernel.evaluate(dataset.data[index], dataset.data[ws])*(new_alpha - old_alpha);
            //if (gradient_change) gradients[index] -= dataset.labels[ws]*dataset.labels[index]*kernel.evaluate(dataset.data[index], dataset.data[ws])*dual_variables[number_of_training_points-1];
            gradients[index] = computeGradient(dataset.data[index], dataset.labels[index], pseudo_variables, pseudo, kernel);
        }
        // Compute approximate gradients for i & j
        //gradients[ws] = computeGradient(dataset.data[ws], dataset.labels[ws], pseudo_variables, pseudo, kernel);
        //gradients[j] = computeGradient(dataset.data[j], dataset.labels[j], pseudo_variables, pseudo, kernel);
        
        //dual_variables[ws] = new_alpha;
        //dual_variables[j] = new_alpha;
        
    }
    double train_end_t = (double)clock() / CLOCKS_PER_SEC;
    //dobjfn_dfile_merging << "epoch :"  << epoch +1 << ":KL:" << KL_max;
    // dobjfn_dfile_merging << "epoch :"  << epoch +1  << ":KL:"  << KL_max<< ":maxPseudogradient:" << maxPseudo << ":mergingtime:" << mergeAndDeleteSV_counter << ":mergingsteps:" << countMerges << endl;
    countMerges = 0;
    
    SVM svm(pseudo_variables, pseudo, kernel);
    // double svmEvaluate = svm.evaluateTestset(testdataset);
    
    dobjfn_dfile_per   << ": per.:"  << svm.evaluateTestset(testdataset)<< ":" << endl;
    
    dobjfn_dfile_traint  << ":trainingtime.:"  << train_end_t - train_start_t << ":" << endl;
    dobjfn_dfile_dualpseudoCounter  << ":dualvarisZero.:"  << dualvarisZero << ":dualvarisC.:"  << dualvarisC << ":dualvarbetZeroC.:"  << dualvarbetZeroC<< ":" << endl;
    dualvarisZero  = 0;
    dualvarisC     = 0;
    dualvarbetZeroC = 0;
    
    
    dobjfn_dfile.close();
    pobjfn_dfile.close();
    
    dobjfn_dfile_param.close();
    pobjfn_dfile_param.close();
    
    dobjfn_dfile_per.close();
    dobjfn_dfile_traint.close();
    dobjfn_dfile_merging.close();
    
    
    return SVM(pseudo_variables, pseudo, kernel);
}
