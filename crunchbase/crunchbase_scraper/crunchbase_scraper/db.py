from sqlalchemy import create_engine, Column, Integer, Table, String, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


engine = create_engine('mysql://admin:admin@localhost/crunchbase?charset=utf8', echo=True)
Base = declarative_base()

class Company(Base):
    __tablename__ = 'companies'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(30))
    crunch_id = Column(String(30))
    website = Column(String(30))
    blog = Column(String(30))
    twitter = Column(String(30))
    category = Column(String(30))
    email = Column(String(30))
    employee_num = Column(Integer)
    founded = Column(String(10))
    desc = Column(String(50))
    #address = Column(String(50))
    #city = Column(String(20))
    #state = Column(String(20))
    #zipcode = Column(String(10))
    #country = Column(String(20))


def main():
    '''
    engine = create_engine('mysql://admin:admin@localhost/crunchbase', echo=True)
    metadata = MetaData(bind=engine)
    
    company_table = Table('companies', metadata,
            Column('id', Integer, primary_key=True),
            Column('name', String(30)),
            Column('crunch_id', String(30)),
            Column('website', String(30)),
            Column('blog', String(30)),
            Column('twitter', String(20)),
            Column('category', String(30)),
            Column('email', String(30)),
            Column('employee_num', Integer),
            Column('founded', String(10)),
            Column('desc', String(50)),
            Column('address', String(50)),
            Column('city', String(20)),
            Column('state', String(20)),
            Column('zipcode', String(10)),
            Column('country', String(20)),
    )

    metadata.create_all()
    '''
    Base.metadata.create_all(engine)

if __name__ == "__main__":
    main()

