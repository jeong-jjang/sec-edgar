from secedgar import CompanyFilings, FilingType
from datetime import date

def get_filings(name_companies, start_date, end_date, download_dir='output'):
    start_date = date(int(start_date[0]), int(start_date[1]), int(start_date[2]))
    end_date = date(int(end_date[0]), int(end_date[1]), int(end_date[2]))
    my_filings = CompanyFilings(cik_lookup=name_companies,
                                start_date=start_date,
                                end_date=end_date,
                                filing_type=FilingType.FILING_10K,
                                count=15,
                                user_agent='Byeonghu Na (wp03052@kaist.ac.kr)')
    my_filings.save(download_dir)

