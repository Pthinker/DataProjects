import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import operator
from db import Company


engine = create_engine('mysql://admin:admin@localhost/crunchbase?charset=utf8')
Session = sessionmaker(bind=engine)

colors = ['#348ABD', '#A60628']

def company_year_bar():
    session = Session()
    numbers = []
    for yr in range(1995, 2013):
        numbers.append(session.query(Company).filter(Company.founded==yr).count())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(range(1995, 2013), numbers, color=colors[0], edgecolor=colors[0], align="center", width=0.8, alpha=0.6, lw=2)
    ax.grid(True)
    ax.set_xlabel('Year')
    ax.set_ylabel('# of founded companies')
    ax.set_title('Number of founded tech companies per year since between 1995 and 2012');
    plt.savefig("plots/company_year_bar.pdf")
    plt.show()

def company_category_bar(year=2012):
    session = Session()
    companies = session.query(Company).filter(Company.founded==str(year))
    category_count = {}
    for com in companies:
        cat = com.category
        if cat is not None:
            category_count[cat] = category_count.get(cat, 0) + 1
    sorted_cat = sorted(category_count.iteritems(), key=operator.itemgetter(1))
    categories = []
    counts = []
    for cat, count in sorted_cat:
        categories.append(cat)
        counts.append(count)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    pos = np.arange(len(categories)) + .5

    ax.barh(pos, counts, color=colors[0], edgecolor=colors[0], align="center", alpha=0.6, lw=2)
    ax.set_yticks(pos)
    ax.set_yticklabels(categories)

    ax.grid(True)

    ax.set_xlabel('# of companies')
    ax.set_ylabel('Category')
    ax.set_title("Tech trend in %d" % year)
    plt.savefig("plots/tech_trend_%d.pdf" % year)
    plt.show()

def company_category_year_plot():
    session = Session() 
    
    category_count = {'web':[], 'software':[], 'mobile':[], 'advertising':[], 'education':[], 'biotech':[]}
    for category in category_count:
        for yr in range(1995, 2013):
            category_count[category].append(session.query(Company).filter(Company.category==category, Company.founded==yr).count())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for category in category_count:
        ax.plot(range(1995, 2013), category_count[category], label=category, lw=2, alpha=0.6)

    ax.legend(loc=2)
    ax.grid(True)

    ax.set_xlabel('Year')
    ax.set_ylabel('# of founded companies')
    ax.set_title("Company category trend")
    plt.savefig("plots/category_trend.pdf")
    plt.show()


def main():
    #company_year_bar()

    #company_category_bar(2012)
 
    company_category_year_plot()


if __name__ == "__main__":
    main()

