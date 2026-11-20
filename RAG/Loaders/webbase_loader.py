from langchain_community.document_loaders import WebBaseLoader

url = 'https://www.flipkart.com/apple-macbook-air-m5-2026-m5-16-gb-512-gb-ssd-tahoe-mdhe4hn-a/p/itm8505e2f874525?pid=COMHZQX47WQ7YJZP&lid=LSTCOMHZQX47WQ7YJZPBZOHQJ&marketplace=FLIPKART&cmpid=content_computer_23776838770_g_8965229628_gmc_pla&tgi=sem,1,G,11214002,g,search,,806047554645,,,,c,,,,,,,&entryMethod=23776838770&&cmpid=content_23776838770_gmc_pla&gad_source=1&gad_campaignid=23776838770&gbraid=0AAAAADxRY5-hZuZNrz7vsidT7VPwo5JJr&gclid=Cj0KCQiA2frXBhC_ARIsAPTFcdMphRU93WojgIRqWOQa5OOcEUqLYdQ_O7OoYjicMGxOveVurUD17SwaAlfFEALw_wcB'
loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))
print(docs[0].page_content)