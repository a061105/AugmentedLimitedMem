#ifndef HASHTABLE
#define HASHTABLE

#include<string>
#include<deque>
#include<vector>
#include<map>
#include<set>

using namespace std;

template<class K, class T>
class HashTable{
	
	public:
	int _capacity;
	int _size;
	//Large Array of vectors
	vector< pair<K,T> >* array;
	set<int> nonzeroIndex;
	
	HashTable(int cap){
		
		_capacity = cap;
		_size =0;
		array = new vector< pair<K,T> >[_capacity];
	}
	
	HashTable(std::istringstream& is, int n){
		
		_capacity = (int)(n*1.5);
		_size =0;
		array = new vector< pair<K,T> >[_capacity];
		
		K k;
		T v;
		for(int i=0;i<n;i++){
			is >> k;
			is >> v;
			this->insert(k,v);
		}
	}
	

	~HashTable(){
		
		delete[] array;
	}
	
	int capacity(){
		return _capacity;
	}

	int size(){
		return _size;
	}
	
	void clear(){

		for(set<int>::iterator it=nonzeroIndex.begin(); it!=nonzeroIndex.end(); it++){
			array[*it].clear();
		}
		_size=0;
		nonzeroIndex.clear();
	}

	void insert(K key,T value){
		
		int h = key % _capacity;
		for(int i=0;i<array[h].size();i++)
			if(array[h][i].first==key){
				
		//		T old_value = array[h][i].second;
				array[h][i].second = value;

		//		return old_value;
			}
		
		array[h].push_back(make_pair(key,value));
		nonzeroIndex.insert(h);
		_size++;
		
		//return NULL;
	}
	
	void remove(K key){
		
		int h = key % _capacity;
		for(int i=0;i<array[h].size();i++)
			if( array[h][i].first == key ){
		//		T old_value = array[h][i].second;
				array[h].erase(array[h].begin()+i);
				if(array[h].size()==0)
					nonzeroIndex.erase(h);
				_size--;
		//		return old_value;
			}
		
		//return NULL;
	}

    T* findval(K key){
		int h = key % _capacity;
		
		for(int i=0;i<array[h].size();i++){
			if(array[h][i].first==key){
				return &(array[h][i].second);
			}
		}
		
		return NULL;
    
    }
	pair<K,T>* find(K key){
		
		int h = key % _capacity;
		
		for(int i=0;i<array[h].size();i++){
			if(array[h][i].first==key){
				return &(array[h][i]);
			}
		}
		
		return NULL;
	}
	
	void keys(vector<K>* v){
		
		v->clear();
		for( set<int>::iterator it=nonzeroIndex.begin();
			it!=nonzeroIndex.end(); it++){
			
			for(int j=0;j<array[*it].size();j++)
				v->push_back( array[*it][j].first );
		}
	}

	void values(deque<T>* v){
		
		v->clear();
		for( set<int>::iterator it=nonzeroIndex.begin();
			it!=nonzeroIndex.end(); it++){
			
			for(int j=0;j<array[*it].size();j++)
				v->push_back( array[*it][j].second );
		}
	}

	HashTable<K,T>& operator=(const HashTable<K,T>& table){
		
		nonzeroIndex = table.nonzeroIndex;
		
		for( set<int>::iterator it = nonzeroIndex.begin();
			it != nonzeroIndex.end(); it++ ){
			
			array[*it] = table.array[*it];
		}
	}

	int max_index(){
		
		if( nonzeroIndex.size() == 0 )
			return 0;

		set<int>::iterator it = nonzeroIndex.end();
		it--;
		typename vector< pair<K,T> >::iterator it2 = array[ *it ].end();
		it2--;
		
		return (int) it2->first;
	}

	
	class iterator{
		
		public:
		set<int>::iterator nnz_it;
		typename vector< pair<K,T> >::iterator v_it;
		HashTable<K,T>* table;
		

		pair<K,T>* operator->() {
			return &(*v_it);
		}

		iterator& operator++(){
			
			v_it++;
			if( v_it == table->array[ *(nnz_it) ].end() ){
				nnz_it++;
				if( nnz_it != table->nonzeroIndex.end() )
					v_it = table->array[ *(nnz_it) ].begin();
			}

			return *(this);
		}

		bool operator==(const iterator& it2) {
			
			return ( nnz_it == it2.nnz_it && v_it == it2.v_it );
		}
		
		bool operator!=(const iterator& it2) {
			
			return !( nnz_it == it2.nnz_it && v_it == it2.v_it );
		}
	};
	
	iterator begin(){
		
		iterator it;
		it.nnz_it = nonzeroIndex.begin();
		it.v_it = array[ *nonzeroIndex.begin() ].begin();
		it.table = this;

		return it;
	}
	
	iterator end(){
		
		iterator it;
		it.nnz_it = nonzeroIndex.end();
		
		it.nnz_it--;
		it.v_it = array[ *(it.nnz_it) ].end();
		it.nnz_it++;
		
		it.table = this;

		return it;
	}
};

#endif
