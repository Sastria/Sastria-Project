from videofacerec import WhoIs

if __name__ == '__main__':
    
    who=WhoIs(-1)
    print who.shelve_db
    for i in range(346):
        if who.shelve_db.has_key(str(i)):
            print i," ", who.shelve_db[str(i)]
        else: 
            print i," ASSENTE "