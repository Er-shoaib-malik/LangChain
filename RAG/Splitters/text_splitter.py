from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """The Apple MacBook Air M5 2026 is a sleek and powerful laptop designed for users who demand high performance and portability. It features the latest M5 chip, which provides exceptional speed and efficiency for all your computing needs. With 16 GB of RAM and a 512 GB SSD, this MacBook Air ensures smooth multitasking and ample storage for your files and applications.
The 13.3-inch Retina display delivers stunning visuals with vibrant colors and sharp details, making it perfect for creative professionals and media consumption. The laptop also boasts a long-lasting battery life, allowing you to work or play for extended periods without needing to recharge.
In terms of design, the MacBook Air M5 2026 maintains its iconic slim and lightweight profile, making it easy to carry around. It also includes a range of connectivity options, including Thunderbolt ports and Wi-Fi 6 support, ensuring fast data transfer and reliable internet connections.
Overall, the Apple MacBook Air M5 2026 is an excellent choice for users seeking a high-performance, portable laptop that combines cutting-edge technology with a sleek design. Whether you're a student, professional, or creative individual, this MacBook Air is built to meet your needs and exceed your expectations.
"""

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

chunks  = splitter.split_text(text)

print(len(chunks))
print(chunks)