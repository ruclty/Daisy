#include <string>
#include <algorithm>
#include <vector>
#include <iostream>
#include <regex>
#include <queue>
#include <map>
#include "querytree.h"
using namespace std;

string query_tree_node::get_str(){
    string ntype = ntype_to_token(node_type);
        if(node_type == PROJECT){
            string proj_cond = "(|";
            for (int i = 0; i < attr_names.size(); i ++){
                proj_cond += table_names[i]+"."+attr_names[i]+"|";
            }
            proj_cond += ")";
            return ntype+proj_cond;
        }
    	if(node_type == JOIN || node_type == SELECT){
        	string predicate = "(|";
        	switch(pred_type){
         	   case INT: for(auto pred : predv)
                         predicate += pred.get_str()+"|";
            	case CHAR: for(auto pred : preds)
                         predicate += pred.get_str()+"|";
            	case TAB: for(auto pred : predt)
                         predicate += pred.get_str()+"|"; 
        	}
        	predicate += ")";
        	return ntype+predicate;
    	}
    	if(node_type == TABLE)
        	return ntype+"("+table_names[0]+")";
    	return ntype;
}

query_tree::query_tree(string& sql) {
	this->sql = sql;
	this->root = NULL;
}

query_tree::~query_tree(){}

void query_tree::split_sql() {
    int sel_index = this->sql.find("select");
    int from_index = this->sql.find("from");
    int where_index = this->sql.find("where");

    this->sel_items = this->sql.substr(sel_index+6, from_index-sel_index-6);
    this->from_items = this->sql.substr(from_index+4, where_index-from_index-4);
    if (where_index > 0)
        this->where_items = this->sql.substr(where_index+5);
}
vector<string> query_tree::parser_regex(string str, string pattern){
    regex e(pattern);
    smatch sm;
    string::const_iterator start = str.begin();
    string::const_iterator end = str.end();
    vector<string> results;
    while(regex_search(start, end, sm, e)){
        string msg(sm[0].first, sm[0].second);
        results.push_back(msg);
        start = sm[0].second;
    }
    return results;
}

void query_tree::parser_sql() {
    vector<string> results;
    //cout << "--------split_sql---------" << endl;
    split_sql();

    //cout << this->sel_items <<"|"<< endl;
    //cout << this->from_items <<"|"<< endl;
    //cout << this->where_items <<"|"<< endl;

    //cout << "---------parser_sel---------" << endl;
    results = parser_regex(sel_items, "[\\w]+\\.[\\w]+");
    for (auto res:results)
        projects.push_back(res);
    //cout << "---------parser_from--------" << endl;
    trim(from_items);
    table_names = split(from_items, ",");
    //cout << "---------parser_where--------" << endl;
    if(where_items.size() != 0){
        results = parser_regex(where_items, "[\\w]+\\.[\\w]+\\=[0-9]+");
        for (auto res:results)
            extract_pred(res, "=", EQ, INT);
        results = parser_regex(where_items, "[\\w]+\\.[\\w]+\\>[0-9]+");
        for (auto res:results)
            extract_pred(res, ">",G, INT);
        results = parser_regex(where_items, "[\\w]+\\.[\\w]+\\<[0-9]+");
        for (auto res:results)
            extract_pred(res, "<",L, INT);
        results = parser_regex(where_items, "[\\w]+\\.[\\w]+\\>=[0-9]+");
        for (auto res:results)
            extract_pred(res, ">=", GE, INT);
        results = parser_regex(where_items, "[\\w]+\\.[\\w]+\\<=[0-9]+");
        for (auto res:results)
            extract_pred(res, "<=", LE, INT);
        results = parser_regex(where_items, "[\\w]+\\.[\\w]+\\!=[0-9]+");
        for (auto res:results)
            extract_pred(res, "!=", NE, INT);
        results = parser_regex(where_items, "[\\w]+\\.[\\w]+\\=\\'[^#]*\\'");
        for (auto res:results)
            extract_pred(res, "=",EQ, CHAR);
        results = parser_regex(where_items, "[\\w]+\\.[\\w]+\\=[\\w]+\\.[\\w]+");
        for (auto res:results)
            extract_pred(res, "=",EQ, TAB);
    }
    /*for (vector<predicateV>::iterator iter = predv.begin(); iter != predv.end(); iter ++){
        cout << iter->get_str() << endl;
    }
    for (vector<predicateS>::iterator iter = preds.begin(); iter != preds.end(); iter ++){
        cout << iter->get_str() << endl;
    }
    for (vector<predicateT>::iterator iter = predt.begin(); iter != predt.end(); iter ++){
        cout << iter->get_str() << endl;
    }*/
}

void query_tree::extract_pred(string str, string tok, RELATION rel, PREDTYPE pred_type){
    int index = str.find_first_of(tok);
    string left = str.substr(0, index);
    string right = str.substr(index+tok.size());
    index = left.find_first_of(".");
    string table = left.substr(0,index);
    string attribute = left.substr(index+1);
    struct predicateV pred1;
    struct predicateS pred2;
    struct predicateT pred3;
    switch(pred_type){
        case INT:
            pred1.rel = rel;
            pred1.value = atoi(right.c_str());
            pred1.table_name = table;
            pred1.attribute = attribute;
            predv.push_back(pred1);
            break;
        case CHAR:
            pred2.rel = rel;
            pred2.table_name = table;
            pred2.attribute = attribute;
            pred2.value = right;
            preds.push_back(pred2);
            break;
        case TAB:
            pred3.rel = rel;
            pred3.left_table = table;
            pred3.left_attr = attribute;
            index = right.find_first_of(".");
            pred3.right_table = right.substr(0, index);
            pred3.right_attr = right.substr(index+1);
            predt.push_back(pred3);
    }
}

void query_tree::generate_tree() {
   // cout << "build join node" << endl;
    map<string, query_tree_node*> table2node;
    for(auto table : table_names){
        query_tree_node* node = new query_tree_node;
        node->node_type = TABLE;
        node->table_names.push_back(table);
        node->parent = NULL;
        table2node[table] = node;
        leaf_table[table] = node;
    }
    query_tree_node* join_root = NULL;
    for(auto pred: predt){
        string left_table = pred.left_table;
        string right_table = pred.right_table;
        query_tree_node* left_node = table2node[left_table];
        query_tree_node* right_node = table2node[right_table];
        if(left_node != right_node){
            query_tree_node* node = new query_tree_node;
            node->node_type = JOIN;
            node->pred_type = TAB;
            node->predt.push_back(pred);
            node->child.push_back(table2node[left_table]);
            node->child.push_back(table2node[right_table]);
            left_node->parent = node;
            right_node->parent = node;
            table2node[left_table] = node;
            table2node[right_table] = node;
            if(join_root == left_node || join_root == right_node || join_root == NULL)
                join_root = node; 
        }
        root = join_root;
    }
    if(root == NULL){
        root = leaf_table[table_names[0]];
    }
    //cout << " build select node" << endl;
    for(auto pred: preds){
        query_tree_node* node = new query_tree_node;
        node->node_type = SELECT;
        node->pred_type = CHAR;
        node->preds.push_back(pred);
        node->parent = NULL;
        node->child.push_back(root);
        root->parent = node;
        root = node;
    }
    for(auto pred: predv){
        query_tree_node* node = new query_tree_node;
        node->node_type = SELECT;
        node->pred_type = INT;
        node->predv.push_back(pred);
        node->parent = NULL;
        node->child.push_back(root);
        root->parent = node;
        root = node;
    }

  //  cout << "build project node" << endl;
    query_tree_node* node = new query_tree_node;
    node->node_type = PROJECT;
    if(projects.empty()){
        node->table_names.push_back("");
        node->attr_names.push_back("*");
    }
    else
        for(auto p : projects){
            int index = p.find_first_of(".");
            node->table_names.push_back(p.substr(0,index));
            node->attr_names.push_back(p.substr(index+1));
        }
    node->parent = NULL;
    node->child.push_back(root);
    root->parent = node;
    root = node;
}

void optimize(query_tree_node* node){}
void optimize(query_tree_node* node1, query_tree_node* node2){}

// print tree
void query_tree::print_tree(){
    cout << "-----------print tree---------" << endl << endl;
    queue<query_tree_node*> q;
    queue<int> deep;
    query_tree_node* last = NULL;
    int last_d;
    q.push(root);
    deep.push(0);
    while(!q.empty()){
        query_tree_node* node = q.front();
        int d = deep.front();
        for(auto n : node->child){
            q.push(n);
            deep.push(d+1);
        }
        if(last != NULL && node->parent != last->parent)
            cout << "||";
        else if(last != NULL)
            cout << " ";
        if(last != NULL && d != last_d)
            cout << endl << endl;
        cout << node->get_str();
        q.pop();
        deep.pop();
        last = node;
        last_d = d;
    }
    cout << endl << endl;
}
//


string query_tree::get_sql(){
    return this->sql;
}
query_tree_node* query_tree::get_root(){
    return this->root;
}
