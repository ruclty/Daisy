#include <string>
#include <algorithm>
#include <vector>
#include <iostream>
#include <tuple>
#include <queue>
#include <map>
#include "../global.h"
using namespace std;

struct query_tree_node{
	NODETYPE node_type;
	
	vector<string> table_names;
	vector<string> attr_names;
	string frag_id;
	vector<string> frag_ids;
	
	PREDTYPE pred_type;
	vector<predicateV> predv;
	vector<predicateS> preds;
	vector<predicateT> predt;
	
	vector<query_tree_node*> child;
	query_tree_node* parent;
	string get_str(void);
};

class query_tree
{
private:
	query_tree_node* root;
	string sql;
	string sel_items="";
	string from_items="";
	string where_items="";
	vector<predicateV> predv;
	vector<predicateS> preds;
	vector<predicateT> predt;
	vector<string> table_names;
	vector<string> projects;
	map<string, query_tree_node*> leaf_table;
	map<string, query_tree_node*> leaf_frag;

public:
	query_tree(string&);
	~query_tree();
	
	void split_sql(void);
	void parser_sql(void);
	vector<string> parser_regex(string, string);
	void extract_pred(string, string, RELATION, PREDTYPE);
	void generate_tree(void);

	void optimize(query_tree_node*);
	void optimize(query_tree_node*, query_tree_node*);

    string get_sql(void);	
	// print tree
	void print_tree(void);

    query_tree_node* get_root(void);
};

