#include "sql_preproccess.h"
#include "QueryOptimise.h"

using namespace std;

int MAIN_SITE_ID = 0

// Operator generate_operator(NODETYPE node_type){
// 	Operator op;
// 	if(node_type == SELECT){
// 		op.opratorIndex = SELECT;
// 		op.sourceTableCount = 
// 		op.sourceTable
// 	} 
// }

string query_plan::semi_join(int target_site, string frag_id1m, string frag_id2){

}

// transfer the sql from source_site_id to target_site_id; need some function from rpc
void transfer_sql(stirng sql, int source_site_id, int target_site_id){

}
// transfer the table data from source_site_id to target_site_id; need some function from rpc
void transfer_table(string table_name, int source_site_id, int target_site_id){

}
// how to transfer to reduce the cost of transferring.
string query_plan::how_to_transfer(string* frag_id1s){

}
// give the frag_id, and get the site this frag on. need some function from etcd;
int find_site(int frag_id){


}
// give the frag_id, and get the table_name of the frag. need some function from etcd;
string find_table_name(int frag_id){


}
// after local site excute the sql, it will generate a new table. get the frag id of this table.
string get_new_frag_id(){

}
// from enum data to string token;
string rela_to_token(RELATION rela){
	swich(rela){
		case EQ:
			return "=";
		case NE:
			return "!=";
		case G:
			return ">";
		case GE:
			return ">=";
		case L:
			return "<";
		case LE:
			return "<=";

	}
}
// process on one tree node;
void query_plan::excute_one_operator(query_tree_node* node, int child_id)
{
	node_type = node.node_type;	

	// select: selectNode to sql,transfer,
	// select * 
	// from table_name(find_table_name from frag_id) 
	// where table_namFe.attribute = ".." or num.

	// child_num = 1
	// attribute_num = 1
	if(node_type == SELECT){
		// get sql
		string result_sql = "select * from ";
		query_tree_node* frag_node = node.child[0];
		string table_name = find_table_name(frag_node.frag_id);
		result_sql += table_name;
        // select * from table_name;
		string predicate_string = " where ";
		predicate_string += table_name;
		predicate_string += '.';
		// select * from table_name where table_name.
		swich(node.pred_type){
			case INT :
				predicate_string += node.predv.atrribute;
				predicate_string += rela_to_token(node.predv.rela);
				predicate_string += to_string(node.predv.value);
			case CHAR :
				predicate_string += node.preds.atrribute;
				predicate_string += rela_to_token(node.preds.rela);
				predicate_string += node.preds.value;
		}
		result_sql += predicate_string;
		result_sql += ';'
		// end of getting sql
		// start to transfer the sql to local site, and get result after it exuted.
		int excute_site = find_site(frag_node.frag_id);
		transfer_sql(result_sql, MAIN_SITE_ID, excute_site);
		result_frag_id = get_new_frag_id();
        // change the node_type from select to fragment.
		query_tree_node* new_node = new query_tree_node();
		new_node.node_type = FRAGMENT;
		new_node.frag_id = result_frag_id;
		*(node.parent.child+child_id) = new_node;
	}

	// project: projectNode to sql,transfer,
	// select attribute
	// from table_name(find_table_name from frag_id)

	// child_num = 1
	// attribute_num = n
	if(node_type == PROJECT){
		// get sql
		string result_sql = "select "
        for(int attributeIndex = 0; attributeIndex < node.attribute_num; attributeIndex++){
        	result_sql += *(node.attribute + attributeIndex);
        	if attributeIndex < node.attribute_num -1{
        	    result_sql += ',';
            }
            else{
            	result_sql += ' ';
            }

        }
        // select at1,at2 ;
        result_sql += 'from ';
        // select at1,at2 from ;
        query_tree_node* frag_node = node.child[0];
        string table_name = find_table_name(frag_node.frag_id);
		result_sql += table_name;
		result_sql += ';';
		// end getting sql
		// start to transfer the sql to local site, and get result after it excuted;
		int excute_site = find_site(frag_node.frag_id);
		transfer_sql(result_sql, MAIN_SITE_ID, excute_site);
		int result_frag_id = get_new_frag_id();
		// change the node type from project to fragment;
		query_tree_node* new_node = new query_tree_node();
		new_node.node_type = FRAGMENT;
		new_node.frag_id = result_frag_id;
		*(node.parent.child+child_id) = new_node;
    }

    // join: joinNode to sql(before excuted ,need to transfer table need join to one site, and need semi-join)
    // select *
    // from table_name1, table_name2
    // where table_name1.attribute = table_name2.attribute;
    if(node_type == JOIN){

    	query_tree_node* left_child = node.child[0];
    	query_tree_node* right_child = node.child[1];
    	
        int left_site = find_site(left_child.frag_id);
        int right_site = find_site(right_child.frag_id);
        string* frag_ids = new string[2];
        frag_ids = &(left_child.frag_id);
        frag_ids+1 = &(right_child.frag_id);
        if (left_site != right_site){
        	// how to transfer: site_plan[0] is source, site_plan[1] is target.
        	string* frag_plan = this.how_to_transfer(left_child.frag_id, right_child.frag_id);
        	// transfer the table need join from source to target site.
        	// get parameters of transfer function.
        	string table_name = find_table_name(frag_plan[0]);
        	int source_site = find_site(frag_plan[0]);
        	int target_site = find_site(frag_plan[1]);
        	//do transfer, and get the new frag id;
        	transfer_table(table_name,source_site,target_site);
        	int new_frag_id = get_new_frag_id();
        	int other_frag_id = frag_plan[0];
        }
        else
        	target_site = left_site;
        	int new_frag_id = left_child.frag_id;
        	int other_frag_id = right_child.frag_id;
        if this.s_join == true
        	string result_frag_id = semi_join(target_site, new_frag_id, other_frag_id);
        else{
        	//get sql(not semi join)
    		string result_sql = "select * from ";
    		string table_name1 = find_table_name(new_frag_id);
	    	string table_name2 = find_table_name(other_frag_id);
	    	result_sql += table_name1;
	    	result_sql += ',';
	    	result_sql += table_name2;
	    	result_sql += " where ";
	    	result_sql += temp_string;
	    	// select * from table_name1,table_name2 where ;
	    	result_sql += table_name1;
	    	result_sql += '.';
	    	result_sql += node.predt.atrribute;
	    	result_sql += rela_to_token(node.predt.rela);
	    	result_sql += table_name2;
	    	result_sql += '.';
	    	result_sql += node.predt.attr_value;
	    	result_sql += ';';
	    	//end getting sql;
	        //get transfer the table to one site;
	    	// start to transfer the sql to local site, and get result after it excuted;
			int excute_site = target_site;
			transfer_sql(result_sql, MAIN_SITE_ID, excute_site);
			int result_frag_id = get_new_frag_id();
        }

        query_tree_node* new_node = new query_tree_node();
		new_node.node_type = FRAGMENT;
		new_node.frag_id = result_frag_id;
		*(node.parent.child+child_id) = new_node;
    }
    
    // 
    if(node_type == UNION){
    	// get sql
    	string result_sql = "select * from ";
    	for(int tableIndex = 0; tableIndex < child_num; tableIndex++){
    		query_tree_node* frag_node = node.child[tableIndex];
    		table_name = find_table_name(frag_node.frag_id);
    		result_sql += table_name;
    		if(tableIndex == child_num -1)
    			result_sql += ";";
    		else
    			result_sql += " union all select * from ";
    	}
    	//end getting sql;
	    //get transfer the table to one site;
	    // start to transfer the sql to local site, and get result after it excuted;
		int excute_site = target_site;
		transfer_sql(result_sql, MAIN_SITE_ID, excute_site);
		int result_frag_id = get_new_frag_id();

		query_tree_node* new_node = new query_tree_node();
		new_node.node_type = FRAGMENT;
		new_node.frag_id = result_frag_id;
		*(node.parent.child+child_id) = new_node;
    }   
}


// hou xu bian li qury tree
void query_plan::excute_query_plan(query_tree_node &node, int child_id)
{
	for(int child_index = 0; child_index < node.child_num; child_index++){
		if(child[child_index].node_type == SELECT or child[child_index].node_type == PROJECT or child[child_index].node_type == JOIN or child[childIndex].node_type == UNION){
			excute_query_plan(child[child_index], child_index)
		}

	// if this node is the leaf node(all childs are table)
	excute_one_operator(nodoe, child_index);
	}
}

// next week's work:
// fill the empty function;
// finish local excute code;