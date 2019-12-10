 

//站点信息数据结构
struct site_info
{
	int site_id;
	vector<int> fragment_ids; 
	vector<int> temp_ids;
    string user;
    string password;
	string ip; 
	string port; 
};

vector<site_info> site[MAX_SITE_COUNT]; 

//表的属性信息数据结构
struct attr_info
{
	string attr_name;
	string type;
	bool is_key = 0;
	int size;
	string dist_name; //属性是如何分布的-分布类型
	double mean;//均值
	double std;//方差
};

//表相关信息的数据结构
struct table_info
{
	string table_name;
	vector<attr_info> attributes; 
	vector<frag_info> fragments; 
};

enum FRAGTYPE{H,V}; //分片划分依据

//分片信息的数据结构
struct frag_info
{	
	string frag_id; 
	vector<attr_info> attributes; 
	int size;
	int site_id; 
    string table_name; 
};

fragment_info fragment [MAX_FRAG_COUNT]; //fragment数组中存放有多少个如此的分片

//写两个函数：通过fragment找site、找table
site_info get_site_info(int site_id){}
frag_info get_frag_info(int frag_id){}
table_info get_table_info(string table_name) {}

