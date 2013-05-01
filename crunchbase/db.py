from sqlalchemy import create_engine, Column, Table, MetaData, ForeignKey
from sqlalchemy import Integer, String, Text, Numeric, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import simplejson as json
import os
import re

engine = create_engine('mysql://admin:admin@localhost/crunchbase?charset=utf8', echo=False)
Base = declarative_base()

class Company(Base):
    __tablename__ = 'companies'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    crunch_id = Column(String(100), nullable=False, index=True, unique=True)
    twitter = Column(String(30))
    category = Column(String(30))
    employee_num = Column(Integer)
    founded = Column(String(10))
    desc = Column(String(200))
    tags = Column(String(150))
    overview = Column(Text)
    total_money_raised = Column(Numeric)
    country = Column(String(20))

class People(Base):
    __tablename__ = 'people'

    id = Column(Integer, primary_key=True)
    crunch_id = Column(String(100), nullable=False, index=True, unique=True)
    first_name = Column(String(50))
    last_name = Column(String(80))
    twitter = Column(String(50))
    tags = Column(String(300))
    overview = Column(Text)

class CompanyPeople(Base):
    __tablename__ = 'company_people'
    
    id = Column(Integer, primary_key=True)
    company = Column(String(100), ForeignKey("companies.crunch_id", ondelete='CASCADE'))
    organization = Column(String(200), ForeignKey("financial_organizations.crunch_id", ondelete='CASCADE'))
    people = Column(String(100), ForeignKey("people.crunch_id", ondelete='CASCADE'), nullable=False)
    is_past = Column(Boolean)
    title = Column(String(100))

class CompanyCompetitor(Base):
    __tablename__ = 'competitors'
    
    id = Column(Integer, primary_key=True)
    company = Column(String(100), ForeignKey("companies.crunch_id", ondelete='CASCADE'), nullable=False)
    competitor = Column(String(100), ForeignKey("companies.crunch_id", ondelete='CASCADE'), nullable=False)

class FinancialOrg(Base):
    __tablename__ = 'financial_organizations'

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    crunch_id = Column(String(200), nullable=False, index=True, unique=True)
    twitter = Column(String(30))
    desc = Column(String(200))
    employee_num = Column(Integer)
    founded = Column(String(10))
    tags = Column(String(300))
    overview = Column(Text)

class Investment(Base):
    __tablename__ = 'investments'

    id = Column(Integer, primary_key=True)
    round_code = Column(String(20))
    amount = Column(Numeric)
    currency = Column(String(10))
    year = Column(String(4))
    month = Column(String(2))
    day = Column(String(2))
    company = Column(String(100), ForeignKey("companies.crunch_id", ondelete='CASCADE'), nullable=False)
    company_source = Column(String(100), ForeignKey("companies.crunch_id", ondelete='CASCADE'))
    people_source = Column(String(100), ForeignKey("people.crunch_id", ondelete='CASCADE'))
    org_source = Column(String(200), ForeignKey("financial_organizations.crunch_id", ondelete='CASCADE'))

def store_companies():
    Session = sessionmaker(bind=engine)
    session = Session()
    
    company_folder = "crunchbase_scraper/crunchbase_scraper/company_json"
    company_files = os.listdir(company_folder)
    for fname in company_files:
        fpath = os.path.join(company_folder, fname)
        with open(fpath) as fh:
            com_dict = json.load(fh, strict=False)
            com = Company()
            com.name = com_dict['name']
            com.crunch_id = os.path.splitext(fname)[0]
            if 'twitter_username' in com_dict:
                com.twitter = com_dict['twitter_username']
            if 'category_code' in com_dict:
                com.category = com_dict['category_code']
            if 'number_of_employees' in com_dict:
                com.employee_num = com_dict['number_of_employees']
            if 'founded_year' in com_dict:
                com.founded = com_dict['founded_year']
            if 'tag_list' in com_dict:
                com.tags = com_dict['tag_list']
            if 'description' in com_dict:
                com.desc = com_dict['description']
            if 'overview' in com_dict:
                com.overview = com_dict['overview']
            if 'total_money_raised' in com_dict:
                money = com_dict['total_money_raised']
                matobj = re.search(r"([\d\.]+)", money)
                if matobj:
                    num = matobj.group(1)
                    if money[-1].upper() == 'M':
                        money = float(num) * 1000000
                    elif money[-1].upper() == 'B':
                        money = float(num) * 1000000000
                    elif money[-1].upper() == 'K':
                        money = float(num) * 1000
                    else:
                        money = float(num)
                    com.total_money_raised = money
            session.add(com)
    session.commit()

def store_people():
    Session = sessionmaker(bind=engine)
    session = Session()
    
    folder = "crunchbase_scraper/crunchbase_scraper/people_json"
    pfiles = os.listdir(folder)
    for fname in pfiles:
        fpath = os.path.join(folder, fname)
        with open(fpath) as fh:
            pdict = json.load(fh, strict=False)
        p = People()
        p.crunch_id = pdict['permalink']
        p.first_name = pdict['first_name']
        p.last_name = pdict['last_name']
        p.twitter = pdict['twitter_username']
        p.tags = pdict['tag_list']
        p.overview = pdict['overview']
        session.add(p)
    session.commit()

def store_companypeople():
    Session = sessionmaker(bind=engine)
    session = Session()
    
    pdict = {}
    for people in session.query(People):
        pdict[people.crunch_id] = 1

    company_folder = "crunchbase_scraper/crunchbase_scraper/company_json"
    company_files = os.listdir(company_folder)
    for fname in company_files:
        fpath = os.path.join(company_folder, fname)
        with open(fpath) as fh:
            com_dict = json.load(fh, strict=False)
        com_id = os.path.splitext(fname)[0]
        for rec in com_dict['relationships']:
            if not rec['person']['permalink'] in pdict:
                continue
            compeo = CompanyPeople()
            compeo.company = com_id
            compeo.is_past = rec['is_past']
            compeo.title = rec['title']
            compeo.people = rec['person']['permalink']
            session.add(compeo)
    session.commit()

    folder = "crunchbase_scraper/crunchbase_scraper/financial_json"
    org_files = os.listdir(folder)
    for fname in org_files:
        fpath = os.path.join(folder, fname)
        with open(fpath) as fh:
            org_dict = json.load(fh, strict=False)
        org_id = org_dict['permalink']
        for rec in org_dict['relationships']:
            if not rec['person']['permalink'] in pdict:
                continue
            compeo = CompanyPeople()
            compeo.organization = org_id
            compeo.is_past = rec['is_past']
            compeo.title = rec['title']
            compeo.people = rec['person']['permalink']
            session.add(compeo)
    session.commit()

def store_competitors():
    Session = sessionmaker(bind=engine)
    session = Session()
    
    company_folder = "crunchbase_scraper/crunchbase_scraper/company_json"
    company_files = os.listdir(company_folder)
    for fname in company_files:
        fpath = os.path.join(company_folder, fname)
        with open(fpath) as fh:
            com_dict = json.load(fh, strict=False)
        com_id = os.path.splitext(fname)[0]
        for rec in com_dict['competitions']:
            comp = CompanyCompetitor()
            comp.company = com_id
            comp.competitor = rec['competitor']['permalink']
            if len(com_id) >= 100:
                print "company:", com_id
            if len(rec['competitor']['permalink']) >= 100:
                print "compe:", rec['competitor']['permalink']
            #session.add(comp)
    session.commit()

def store_financial_organizations():
    Session = sessionmaker(bind=engine)
    session = Session()
    
    folder = "crunchbase_scraper/crunchbase_scraper/financial_json"
    org_files = os.listdir(folder)
    for fname in org_files:
        fpath = os.path.join(folder, fname)
        with open(fpath) as fh:
            org_dict = json.load(fh, strict=False)
        org = FinancialOrg()
        org.name = org_dict['name']
        org.crunch_id = org_dict['permalink']
        org.twitter = org_dict['twitter_username']
        org.desc = org_dict['description']
        org.employee_num = org_dict['number_of_employees']
        org.founded = org_dict['founded_year']
        org.tags = org_dict['tag_list']
        org.overview = org_dict['overview']
        session.add(org)
    session.commit()

def store_investments():
    Session = sessionmaker(bind=engine)
    session = Session()
    
    folder = "crunchbase_scraper/crunchbase_scraper/company_json"
    company_files = os.listdir(folder)
    for fname in company_files:
        fpath = os.path.join(folder, fname)
        with open(fpath) as fh:
            com_dict = json.load(fh, strict=False)
        for funding in com_dict['investments']:
            invest = Investment()
            rec = funding['funding_round']
            invest.round_code = rec['round_code']
            invest.amount = rec['raised_amount']
            invest.currency = rec['raised_currency_code']
            invest.year = rec['funded_year']
            invest.month = rec['funded_month']
            invest.day = rec['funded_day']
            invest.company = rec['company']['permalink']
            invest.company_source = com_dict['permalink']
            session.add(invest)
    session.commit()

    folder = "crunchbase_scraper/crunchbase_scraper/people_json"
    people_files = os.listdir(folder)
    for fname in people_files:
        fpath = os.path.join(folder, fname)
        with open(fpath) as fh:
            p_dict = json.load(fh, strict=False)
        for funding in p_dict['investments']:
            invest = Investment()
            rec = funding['funding_round']
            invest.round_code = rec['round_code']
            invest.amount = rec['raised_amount']
            invest.currency = rec['raised_currency_code']
            invest.year = rec['funded_year']
            invest.month = rec['funded_month']
            invest.day = rec['funded_day']
            invest.company = rec['company']['permalink']
            invest.people_source = p_dict['permalink']
            session.add(invest)
    session.commit()
    cdict = {}
    for com in session.query(Company):
        cdict[com.crunch_id] = 1

    folder = "crunchbase_scraper/crunchbase_scraper/financial_json"
    org_files = os.listdir(folder)
    for fname in org_files:
        fpath = os.path.join(folder, fname)
        with open(fpath) as fh:
            org_dict = json.load(fh, strict=False)
        for funding in org_dict['investments']:
            rec = funding['funding_round']
            if not rec['company']['permalink'] in cdict:
                continue
            invest = Investment()
            invest.round_code = rec['round_code']
            invest.amount = rec['raised_amount']
            invest.currency = rec['raised_currency_code']
            invest.year = rec['funded_year']
            invest.month = rec['funded_month']
            invest.day = rec['funded_day']
            invest.company = rec['company']['permalink']
            invest.org_source = org_dict['permalink']
            session.add(invest)
    session.commit()

def create_tables():
    Base.metadata.create_all(engine)


def main():
    create_tables()

    #store_companies()
 
    #store_financial_organizations()       
    
    #store_people()

    #store_competitors()

    #store_companypeople()

    store_investments()


if __name__ == "__main__":
    main()

