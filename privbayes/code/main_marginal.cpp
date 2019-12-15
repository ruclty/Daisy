#include <iostream>
#include <fstream>
#include <tgmath.h>
#include <cmath>
#include <cstdio>

using namespace std;

#include "methods.h"

double exp_marginal(table& tbl, base& base, int k) {
	const vector<vector<int>> mrgs = tools::kway(k, tools::vectorize(tbl.dim));
	double err = 0.0;
	for (const vector<int>& mrg : mrgs) err += tools::TVD(tbl.getCounts(mrg), base.getCounts(mrg));
	return err / mrgs.size();
}

int main(int argc, char *argv[]) {
	// arguments
	string dataset = argv[1];
	cout << dataset << endl;

	int rep = stoi(argv[2]);

	vector<double> thetas;
	for (int i = 3; i < argc; i++) {
		thetas.push_back(stod(argv[i]));
		cout << thetas.back() << "\t";
	}
	cout << endl;
	// arguments

//cout<<"end of juno" <<endl;

	ofstream out("results/" + dataset + ".out");//out contains out
	ofstream log("results/" + dataset + ".log");//log contains cout
	cout.rdbuf(log.rdbuf());

//	cout<<"end of juno1" <<endl;

	random_device rd;						//non-deterministic random engine
	engine eng(rd());						//deterministic engine with a random seed

//	cout<<"end of juno2" <<endl;

	table tbl("data/" + dataset, true);
	vector<int> queries = { 2, 3 };
	
	ofstream ndata("results/"+dataset+".txt");
	for (int i = 0; i < tbl.data.size();i ++){
		for (int d : tbl.data[i])
			ndata << d << ",";
		ndata << endl;
	}			
	ndata.close(); 

//	cout<<"end of juno3" <<endl;

	for (double theta : thetas) {
		cout << "theta: " << theta << endl;
		out << "theta: " << theta << endl;
		for (double epsilon : {100}) {
			vector<double> err(queries.size(), 0.0);
			for (int i = 0; i < rep; i++) {
				cout << "epsilon: " << epsilon << " rep: " << i << endl;
				bayesian bayesian(eng, tbl, epsilon, theta);
				
				string fname = dataset;
				ofstream sample("results/"+fname+"_"+to_string(epsilon)+"_"+to_string(i)+".csv");
				table syn = bayesian.syn;
				//sample << "col0,col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15" << endl;
				for (int i = 0; i < syn.data.size();i ++){
					for (int j = 0; j < syn.data[i].size();j ++){
						int d = syn.data[i][j];
						sample << syn.translators[j]->int2str(d);
						//sample << d;
						if (j != syn.data[i].size()-1)
							sample << ",";
					}
					sample << endl;
				}			
				sample.close();
				
				//for (int qi = 0; qi < err.size(); qi++) {
				//	err[qi] += exp_marginal(tbl, bayesian, queries[qi]);
				//}
			}
			
			//for (int qi = 0; qi < err.size(); qi++) {
			//	out << err[qi] / rep << "\t";
			//}
			out << endl;
		}
		cout << endl;
		out << endl;
	}
	out.close();
	log.close();
//        cout<<"end of juno" <<endl;
	return 0;
}
