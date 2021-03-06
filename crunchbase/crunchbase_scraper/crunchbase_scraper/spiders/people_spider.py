from scrapy.spider import BaseSpider
from scrapy.selector import HtmlXPathSelector
from scrapy.http import Request
import simplejson
import os


class FinancialOrgSpider(BaseSpider):
    name = "people"
    allowed_domains = ["crunchbase.com"]
    api_key = "sn86p65fg2m5cwu6gjtz7dep"

    clist = map(chr, range(97, 123))
    clist.append('other')
    start_urls = []
    for c in clist:
        start_urls.append("http://www.crunchbase.com/people?c=%s" % c)

    def parse(self, response):
        hxs = HtmlXPathSelector(response)
        org_urls = hxs.select('//table[@class="col2_table_listing"]//li/a/@href').extract()

        for url in org_urls:
            crunch_id = url.split("/")[-1].strip()
            api_url = "http://api.crunchbase.com/v/1/person/%s.js?api_key=%s" % \
                    (crunch_id, self.api_key)
            yield Request(api_url, callback=lambda r, crunch_id=crunch_id:self.parse_json(r, crunch_id))

    def parse_json(self, response, crunch_id):
        fpath = "crunchbase_scraper/people_json/%s.json" % crunch_id
        if not os.path.exists(fpath):
            with open(fpath, "w") as fh:
                fh.write(response.body)

