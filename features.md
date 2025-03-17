Key Features to Implement

A. Investment Policy Statement (IPS)
- [ ] 1. Support rendering Investment Policy Statement (IPS) as the main page. This should be the index page.
- [ ] 2. Users should be able to write the IPS if not already set. There should be a sign button, once clicked, user can enter the date and name as signature. The IPS should be saved as a markdown file.
- [ ] 3. Afterwards, the IPS cannot be directly edited. Unless user click c red "Adjust My IPS" button, which pops a warning.

B. Efficient Frontier Calculation
- [ ] 1. The EF expected return and standard deviation should use category average value over the years. The specified funds should be used as representative of each category. Data of each ticker should be retrieved using Yahoo Finance API, using yfinance python package. The data retrieval should be implemented in stock_handler.py and referenced in app.py

        # Equity funds
        self.equity_large_value = "SWLVX"  # Schwab U.S. Large-Cap Value Index Fund
        self.equity_large_blend = "SWTSX"  # Schwab Total Stock Market Index Fund
        self.equity_large_growth = "FSPGX"  # Fidelity® Large Cap Growth Index Fund
        
        self.equity_mid_value = "SWMCX"  # Schwab U.S. Mid-Cap Index Fund
        self.equity_mid_blend = "VEXAX"  # Vanguard Extended Market Index Fund Admiral Shares
        self.equity_mid_growth = "IJK"  # iShares S&P Mid-Cap 400 Growth ETF
        
        self.equity_small_value = "VBR"  # Vanguard Small-Cap Value ETF
        self.equity_small_blend = "SWSSX"  # Schwab Small-Cap Index Fund
        self.equity_small_growth = "FCPGX"  # Fidelity Small Cap Growth
        
        # Fixed income funds
        self.fixed_short_low = "FSHBX"  # Fidelity® Short-Term Bond Fund (Investment Grade)
        self.fixed_short_mid = "SJNK"  # SPDR® Bloomberg Short Term High Yield Bond ETF (High Yield/Junk)
        self.fixed_short_high = "ISTB"  # iShares Core 1-5 Year USD Bond ETF (U.S. Government)
        
        self.fixed_mid_low = "FBNDX"  # Fidelity® Investment Grade Bond Fund
        self.fixed_mid_mid = "LSYAX"  # Lord Abbett Short Duration High Yield Fund
        self.fixed_mid_high = "VGIT"  # Vanguard Intermediate-Term Government Bond ETF
        
        self.fixed_long_low = "VCLT"  # Vanguard Long-Term Corporate Bond ETF
        self.fixed_long_mid = "JNK"  # SPDR® Bloomberg High Yield Bond ETF
        self.fixed_long_high = "VGLT"  # Vanguard Long-Term Government Bond ETF
        

C. Render Representative fund statistics

- [ ] 1. Each fund in stock_handler.py should have it's data temporarily saved in memory.
- [ ] 2. There should be fund stats page ("Fund Stats") that allows user to choose which fund's stats to render
- [ ] 3. The page should just render yahoo finance page directly. i.e https://finance.yahoo.com/quote/FCPGX/ for FCPGX