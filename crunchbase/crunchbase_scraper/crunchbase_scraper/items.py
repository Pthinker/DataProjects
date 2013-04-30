# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/topics/items.html

from scrapy.item import Item, Field

class CompanyItem(Item):
    name = Field()
    crunch_id = Field()
    website = Field()
    blog = Field()
    twitter = Field()
    category = Field()
    email = Field()
    employee_num = Field()
    founded = Field()
    desc = Field()
    #address = Field()
    #city = Field()
    #state = Field()
    #zipcode = Field()
    #country = Field()

