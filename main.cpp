#include <string>
#include <algorithm>
#include <vector>
#include <iostream>
#include "querytree/querytree.cpp"
using namespace std;

int main(){
    
    string sql;
    while(getline(cin, sql)){
        query_tree tree(sql);
        tree.parser_sql();
        tree.generate_tree();
        tree.print_tree();
    }
    return 0;
}
