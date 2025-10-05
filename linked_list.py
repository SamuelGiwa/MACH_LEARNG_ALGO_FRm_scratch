class node:
    def __init__(self, data):
        self.data = data
        self.next = None   # pointer to the next node


class linkedlist:
    def __init__(self, length):
        self.length = length
        self.head = None   # start of the list

    def node_(self):
        """Create a linked list of given length with user input."""
        for i in range(self.length):
            val = input(f"Enter data for node {i+1}: ")
            new_node = node(val)
            if self.head is None:    # first node
                self.head = new_node
            else:
                temp = self.head
                while temp.next:     # move to the end
                    temp = temp.next
                temp.next = new_node
                
                
    def display(self):
        """Print the linked list."""
        temp = self.head
        while temp:
            print(temp.data, end=" -> ")
            temp = temp.next
        print("None")
        
    def insert(self):
        pass
    
    def delete(self,pos):
        pass
        
    
    def search(self, item_, by_index = False):
        x = self.head
        index = 0
        if by_index == True:
            while x is not None:        
                if int(x.data ) == item_:
                    return index     # found it
                x = x.next
                index += 1
            return -1  
        else:         
            while x is not None:        
                    if int(x.data )== item_:
                        print("you are a man")
                        return index     # found it
                    x = x.next
                    index += 1
            return -1 
        
        
              

ll = linkedlist(3)
ll.node_()
ll.display()
print(ll.search(2, by_index=True))