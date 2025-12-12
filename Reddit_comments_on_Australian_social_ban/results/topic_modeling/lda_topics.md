## Topic 0: Politics & Governance
**Post 1:**

> simple guide for privacy # Edits and Updates

**a list of eddits and dates to keep track of what changed from the original post.**

**28/11/2023**

* thanks to KrazyKirby i have an important note wich will also be placed in the hardening Firefox section
* Made an edit on the Brave information due to details shared by Accomplished\_Shoe
* Thanks to 007008009010, Slashtab and Bettervanilla for giving the amazing ideas to make the guide more easy to use and navigate

**29/11/2023**

* moved Google down to Extreme Spyware due to a new update wich is in development
* made a mention to a video explaining the reason for moving down google from spyware to extreme spyware
* shortened the links for ease of use
* Deleted the mention of Brave in the browser section
* updated the Tor description
* Removed the Small question section
* Cant name the acronym for Vitrual Private Network in the Tor section, this sub wont allow that. so i used the full name instead to bypass it, hopefully that wont cause this guide to get taken down

**30/11/2023**

* added a new section: extensions

**16/12/2023**

* removed "i dont care about cookies" from the extension list, due to it being owned by data harvesters

**17/12/2023**

* added "i still dont care about cookies" to the extension list
* fixed some issues with the guide due to me adding and removing stuff

**18/12/2023**

* added a Cromite to android browsers
* removed the duckduckgo browser from pc browsers
* updated description on ungoogled chromium

**21/12/2023**

* removed the mention of using tor with a v\*n running, since thats not reccomended, could break tors use for being anonymous and its "hoax"
* made a note on the Tor browser recomendation that you shouldnt use a v\*n when using Tor
* removed Duckduckgo from the TL;DR search engine section

**27/12/2023**

* updated the description of SearX
* removed the following extentions from the list: Private Bookmarks, Search by image
* removed DuckDuckGo from the search engine section

**03/03/2024**

* removed Skiff since it is sadly shutting down in the comming months

**25/03/2024**

* fixed 1 of the searx instances not being noted

**14/04/2024**

* swapped Proton Pass out for KeepassXC
* moved the edit / update list to the top of the guide. it makes more sense this way so. instead of scrolling all the way down, you can now quickly find the updates

**26/07/2024**

* added changed / added some extra information to the Startpage Entry

**17/09/2024**

* removed the spyware category in my browser section and combined it with the Extreme Spyware category
* added Brave to the extreme spyware category with a megathread with all the details on why people dont trust it

# Side Node

the original post i made got taken down by one of the moderator bots due to a lot of links to youtube videos for explanations. i deleted the links to videos so hopefully this guide will stay up.

This is a guide that most people will (hopefully) find usefull.

# Content List

in this guide i will cover the following things:

* Best Mobile (Ios and Android) Browser
* Best Pc browser (windows, mac and Linux)
* Best Email
* Best search Engine
* a guide on how to harden firefox
* a TL;DR (To Long, Didnt Read) section
* extensions
* closing notes
* Edits section

# Browsers

**IOS**

on IOS, all browsers are basically the same.

why is this? this is because Apple, the main devs behind IOS, force browsers to use Webkit.

this makes all browsers act the same like Safari. so for ios, there really is no real reccomendation i can give, since all browsers are the same aside from Name and Logo.

**Android**

* **Mull** - Mull can be found on the Fdroid app and is hardened firefox but on your phone. it supports extentions, but not everything. extensions are also not fully compatible with mobile, since they werent intended for that. do note that Mull might have performance issues, wich can be annoying.
* **Tor -** the undeniable best browser if you wanna be private on the internet. its important to change some settings for the absolute best privacy. tor on android is on the slower side and its not really reccomended to use as a daily browser
* **Cromite** - Cromite is a fork of Bromite, a now dead browser. its a chromium based browser wich means it has per site isolation (something that mull and tor dont have according to this [source](https://divestos.org/pages/browsers). Cromite is Fast, has a build in addblocker (although uBo is still better) and automatically declines third party cookies (although you still get the cookie requests). all in all, an excelent browser for day to day use

**PC**

PC browsers will focus more info and will be ranked in tiers:

**(Extreme) Spyware**

* Opera and its gaming variant OperaGX. Opera and its little brother OperaGX are both owned by a chinese company wich collects your data, logs your adress and more. its like tiktok, but in a browser form and everyone promoting it doenst know about the dark truth behind it.
* Firefox.i know that it sounds controversial right of the bat, but let me explain. if you dont change any settings on firefox, data will be collected and sent to both Mozilla and Google wich we dont want.thats the main reason why Firefox is here. Luckely, Firefox has an extremely good way to fix this wich we will discuss later.
* Google: do i really need to explain this one? Google is well known for their data harvesting and targeted adds. hell, its so famous there are even videos of people testing if google is listening to them for targeted adds.there is also Chromium, but i dont have experience with that so i cant judge on it, luckely there is another option wich is google but better wich we will discuss later.
* **29/11/2023 edit**
* thanks to a video of Mutahar (SomeOrdinaryGamers) i have been informed that google is making changes so more data is routed to them. this will first be optional, but later on in the future will become default. check out the video for more information. the video in question can be found [here](https://www.youtube.com/watch?v=0Qqjn38TV6g)

* Microsoft Edge (and in the past Internet Explorer): the same story as google, but instead of sending your data to google its sent to Microsoft.
* Brave, it has had many controversies over the years since its release and luckely there is a mega thread with all the information you can find and read [here](https://www.reddit.com/r/privacy/comments/ofnnlb/brave_browser_is_it_as_unsecure_as_the_firefox/)

**Amazing Browsers**

* Ungoogled Chromium: Ungoogled chromium is the chromium browser, but all ties to google are removed and blocked. this does break some things, like adding extensions to your browser. the design of ungoogled chromium is very close to google, so if you want to get rid of google but dont want to switch to a different UI (User Interface) like Firefox, then this is the browser for you.

**Tied Second Place Browsers**

* Librewolf: Librewolf is a firefox fork wich automatically hardens firefox for you. Librewolf uses a custom version of the Arkenfox Project to harden there browser.Librewolf is excelent for people who want hardened firefox, but dont know how to harden it / have issues with it.Hardened Firefox:
* Hardened firefox is (along with librewolf) are the tied second place best browsers.there are 3 populair scritps for firefox to harden it, depending on what you need.
* Easy mode: Betterfox. good privacy and good security.
* Advanced mode: Arkenfox, high privacy and high securityExpert mode.
* Narsils User Js, this disables almost all settings for the highest privacy and security.
* i will link the User.js files in the Hardening section.

**Tor**

* and now, for the number one browser for privacy: Tor. Tor is based on Hardened firefox and has the best privacy a browser can give.Tor also comes with the added benefit of being able to access the Deep and Dark web, but if you want to do that be carefull. Tor also has 2 rules to always follow:

1. Never log in on any website while using Tor. if you do, then the whole purpose of Tor is just gone. If you really want to log in on Tor, make a new account specifically for Tor and only use the account on Tor.
2. NEVER use a v\*n with tor. if you do this, you will stick out and you will no longer be anonymous. if you do want to use a v\*n for safety, start it before opening tor. but you generally [shoudlnt](https://www.reddit.com/r/TOR/comments/5qmqw4/should_i_be_using_a_vpn_with_tor/).

# Best Email

**Best Email**

* Proton: end to end encryption for both Proton and Non proton users. if you send a mail to someone who isnt using Proton, you can encrypt your mail with a password.

# Search Engine

**Best Search Engine**

* Startpage: private search engine wich is now owned by an add company. luckely this hasnt changed anything about there amazing search engine wich uses both the Google and Bing index (its barely noticable, unless you search super specific things and look at pictures related to the search)
* SearX: SearX or now known as SearXng is by far the best search engine for privacy and amazing results. its open source and is hosted by multiple people and companies. you can also self host. there are things you should know before using SearX tho: if you want to use an instance (someones puplic server to do your searches), you only should use 2 of them:

1. [searx.be](https://searx.be/) this is the default instance that Librewolf offers as a search engine option. this is also one of the servers that is always up and is reccomended by most people.
2. [https://search.disroot.org/](https://search.disroot.org/), disroot is a company, wich is good for there own instance since they have a reputation they need to keep up. the only downside of the disroot instance is that its always targeted by people trying to breach it due to them being political (atleast, thats what i heard)

* but what if you dont trust puplic instances, what then? you can self host your own server, wich opens up the possibility to host your own instance for others, a select group of people or only just you. do know that unless there is a big crowd on an instance, you will easily be identified since your searches are sent to google and such. if you are the only one, or there are less than 100 people on an instance its easier for them to tie the searches back to you.
* luckely, there is a way to make you anonymous with searches on SearX. you need to host your own instance with a proxy enabled. this removes your ip from searches and only gives google and others your searches.
* i dont know if there is a solution to also remove the searches from being send out, but its alteast better than your ip being given to them along with your searches.
* i also dont have any guides to set up a server, so you have to do your own digging.
* lastly, searx might have performance issues, due to it requesting results from other engines you enable. in my eyes, this is a small nitpick for an amazing engine.

# How to harden FireFox

before we can harden firefox, we first need to identify your needs.

what do i mean? what do you want from hardening firefox?

if you want maximum privacy and security, use Narsils js or use tor.

want decent privacy and security, use Betterfox.

for the best of both worlds, use Arkenfox.

hardening your firefox also will change depending on your thread model.

what is a thread model? a thread model is more or less your goals when it comes to being private and secure online.

i reccomend watching a video from Eric Murphy on more info on a Thread Model. the video is called "Nobody understands online". he has amazing videos, tierlists for browsers and more. i can highly reccomend his channel for information and guides.

found your thread model? good, now lets begin.

download firefox along with the user.js wich fits your needs.

here are the 3 most populair ones:

[Betterfox](https://github.com/yokoffing/Betterfox)

[Arkenfox](https://github.com/arkenfox/user.js/)

[Narsils js](https://codeberg.org/Narsil/user.js/src/branch/main/desktop)

Note: some of the user.js files require reading like Arkenfox, do this to get a good understanding on how hardening works, features it has and how to change some.

after installing firefox, open it and enter "about:profiles" in your search bar.

click enter and you will be brought to a profile page.make a new profile and call it the same as your User.js (optional, but i prefer to do this. its also handy if you run multiple profiles with different user.js files).

click "select as default" on your new profile and close firefox.

restart firefox and delete the non used profile for ease of use.

get your User.js file, copy paste it and navigate to %appdata% > roaming > mozilla > firefox > profiles > your profile you just made and paste your user js in here.

great. now firefox has its user.js installed and its hardened.however, where not done yet.

when installing your User.js, there could be an updater.sh (or .bat if your on windows) script along with a prefcleaner.sh (or .bat on windows) wich was also downloadable on the user.js file page.

drop these 2 files (sh on linux and bat on windows, not both) in the same profile box as your user.js.

the updater script will get the latest update from the github page of your user.js and update your current one.

the prefs cleaner is to wipe all your firefox settings (i dont reccomend using this).

IMPORTANT NOTE: as KrazyKirby pointed out:

oh no don't do this.. that's not how you're supposed to use it, and you could stuck with that value forever (which could break your browser or cause vulnerability)you're suppose to 1. copied all of arkenfox files to your profile directory 2. create user-override.js (name may be wrong, please check actual arkenfox how to) 3. run the arkenfox script 4. run the script again every time you updated firefox (also run pref cleaner every time) you may also need to read up arkenfox update to re-custom your override script tootl;dr DON'T JUST SHOVE user.js INTO YOUR PROFILE

now, since the browser likely wont fully function or has things that you want changed, you can make an user-override.js file where you can make a command list to change things inside your browser.

do note, after making your overrides file and saving, you need to run your updater script, say start and then enter y if you want to do a preference cleaner afterwards or n if you want to keep your firefox settings.

if your using Arkenfox or any of the other user.js files, some settings on restart of the browser will change.

a couple can be fixed with the override file mentioned above, but there are a couple i have no fix for: these include setting your dns security to max protection with either NextDNS or a custom one and forcing firefox to always ask where your downloaded files need to be saved along with saving other files right under this option.

# TL;DR

**a summary of the results**

**Browsers**

* Brave for IOS
* Mull for Android
* Librewolf or Hardened FireFox for pc

**Mail**

* proton
* skiff

**Search Engine**

* Startpage
* SearX

**Hardened Firefox**

i cant really make this shorter, sadly.

# Extensions

Getting extentions for your browser will make your browsing experience WAY better and safer. some are useless or can be switched for other extensions wich do the job allready / way better.

the extensions will be put into tiers depending on what they do.

all extention names will link to there respective page to add them to your browser.

(**Note:** all links used are from the Mozilla Addon website, the links might not work for other browsers)

**uBlock**

* [uBlock Origin](https://addons.mozilla.org/en-US/firefox/addon/ublock-origin/): the best "Add" blocker extension there is. why do i have quotes around add? because uBlock blocks more than just adds. it blocks cookies, youtube adds, popups and much more. Ublock also features "modes" wich change how strict ublock behaves. the modes are:

1. Very Easy (Standard settings when you install Ublock)
2. Easy (change a bit of the settings from standard mode)
3. Medium (the reccomended blocking mode, will break sites tho)
4. Hard (Extreme blocking, breaks the web)

for a guide on how to enable different blocking modes, along with the difference between each mode click [here](https://github.com/gorhill/uBlock/wiki/Blocking-mode)

**Blockers**

* [Grease Monkey](https://addons.mozilla.org/en-US/firefox/addon/greasemonkey/): Grease Monkey by itself does nothing. Grease monkey is used as a script executer. there are multiple scripts you could use, the 3 i'm using are:
* [Return Youtube Dislikes](https://www.returnyoutubedislike.com/install) - brings back YouTube dislikes
* [Anti Addblocker](https://reek.github.io/anti-adblock-killer/#filterlist) - blocks websites from trying to disable your addblocker, in other words a script to stop add block detection
* [YouTube age verification bypass](https://greasyfork.org/en/scripts/423851-simple-youtube-age-restriction-bypass) - the name explains itself. doenst require logging in or verification
* [i still dont care about cookies](https://github.com/OhMyGuus/I-Still-Dont-Care-About-Cookies) - i still dont care about cookies is a fork of the "i dont care about cookies" extension, because the original is now owned by Avast (a data harvesting company). this extension can be found on github, has nice install guide and is secure

**Password Managers**

* [Bitwarden](https://addons.mozilla.org/en-US/firefox/addon/bitwarden-password-manager/): the most populair password manager out there, and for good reason. i am still in the process of switching, because i use both Bitwarden and another Password Manager.
* [KeepassXC](https://keepassxc.org/): its a seperate app that stores passwords locally (meaning you are the only one that can access them as long as you have the passwords file and remember the master password to access your database) but also has browser integration.

**Qol (Quality Of Life)**

* [Mazimize All Windows](https://addons.mozilla.org/en-US/firefox/addon/maximize-all-windows/): if you have either Librewolf or hardened firefox, your browser will always open in windowed mode (both for normal and private browsing). this extension fixes that by automattically changing the browser size from Windowed to Fullscreen.
* [Dark Reader](https://addons.mozilla.org/en-US/firefox/addon/darkreader/): by far the best extension for your eyes. Dark reader forces all sites to show in a dark theme (there are exceptions, but those are rare). this is nice when you browse the web at later hours. you can finally browse the web without getting flashbanged constantly.
* **Note**: an extra step you can take to make using any device better for your eyes, is turning on Night Light and setting it to 50%. Night Light removes Blue Light, wich is extremely bad for your eyes (i have glasses myself because of this) if you set the Night Light to 50%, you get 50% less blue light wich makes the colours on your device way softer and nicer for your eyes to look at. give it a try.
* [Song Identifier](https://addons.mozilla.org/en-US/firefox/addon/song-identifier/): ever heard a song somewhere on the internet and you cant figure out the name of it? ye, i had that happen alot of times. luckely this extension does the work for you. not every song can be identified, but most of the time it does. it might not be as good as Shazam, but it will do the trick.

**YouTube Qol**

* [Unhook](https://addons.mozilla.org/en-US/firefox/addon/youtube-recommended-videos/): this extension allows its users to customize YouTube to what you want. dont want the shorts tab? you can disable that. you want the comments removed? can be done. want the entire homepage to only show videos from your subscribed channels? guess what, you can do that.
* [Sponsor Block](https://addons.mozilla.org/en-US/firefox/addon/sponsorblock/): since we already are blocking adds with add blockers, how about we get rid of those irritating sponsors? Using Sponsor Block, you can stop:

1. Sponsers, Duh
2. Unpaid self promotion, blocks mentions of merch and things like that
3. Exclusive access, content that you can only watch if you pay for a channels membership or for there Patrion, SubscribeStar etc.
4. Intro's, handy if you directly want to get to the content
5. Endcards / Credits, skips credits altough most people will probably click off videos before this kicks in
6. and much more!

# Closing Notes

thats the end of the guide, hopefully it was usefull. please feel free to correct me if i was wrong or give other reccomendations.

if you need any more info or links, just ask and i will do my digging
**Post 2:**

> what are your thoughts on Chatgpt's updated Terms of Use and Privacy Policy Effective: December 14, 2023

﻿We’ve updated our Privacy Policy below. These updates do not apply to individuals located in the European Economic Area, UK, and Switzerland. If you reside in those areas, the previous version of our Privacy Policy continues to apply to you.

We at OpenAI OpCo, LLC (together with our affiliates, “OpenAI”, “we”, “our” or “us”) respect your privacy and are strongly committed to keeping secure any information we obtain from you or about you. This Privacy Policy describes our practices with respect to Personal Information we collect from or about you when you use our website, applications, and services (collectively, “Services”). This Privacy Policy does not apply to content that we process on behalf of customers of our business offerings, such as our API. Our use of that data is governed by our customer agreements covering access to and use of those offerings.

For information about how we collect and use training information to develop our language models that power ChatGPT and other Services, and your choices with respect to that information, please see this help center article.

1. Personal information we collect
We collect personal information relating to you (“Personal Information”) as follows:

Personal Information You Provide: We collect Personal Information if you create an account to use our Services or communicate with us as follows:

Account Information: When you create an account with us, we will collect information associated with your account, including your name, contact information, account credentials, payment card information, and transaction history, (collectively, “Account Information”).
User Content: When you use our Services, we collect Personal Information that is included in the input, file uploads, or feedback that you provide to our Services (“Content”). 
Communication Information: If you communicate with us, we collect your name, contact information, and the contents of any messages you send (“Communication Information”).
Social Media Information: We have pages on social media sites like Instagram, Facebook, Medium, Twitter, YouTube and LinkedIn. When you interact with our social media pages, we will collect Personal Information that you elect to provide to us, such as your contact details (collectively, “Social Information”). In addition, the companies that host our social media pages may provide us with aggregate information and analytics about our social media activity.
Other Information You Provide: We collect other information that you may provide to us, such as when you participate in our events or surveys or provide us with information to establish your identity (collectively, “Other Information You Provide”
Personal Information We Receive Automatically From Your Use of the Services: When you visit, use, or interact with the Services, we receive the following information about your visit, use, or interactions (“Technical Information”):

Log Data: Information that your browser or device automatically sends when you use our Services. Log data includes your Internet Protocol address, browser type and settings, the date and time of your request, and how you interact with our Services.
Usage Data: We may automatically collect information about your use of the Services, such as the types of content that you view or engage with, the features you use and the actions you take, as well as your time zone, country, the dates and times of access, user agent and version, type of computer or mobile device, and your computer connection.
Device Information: Includes name of the device, operating system, device identifiers, and browser you are using. Information collected may depend on the type of device you use and its settings.
Cookies: We use cookies to operate and administer our Services, and improve your experience. A “cookie” is a piece of information sent to your browser by a website you visit. You can set your browser to accept all cookies, to reject all cookies, or to notify you whenever a cookie is offered so that you can decide each time whether to accept it. However, refusing a cookie may in some cases preclude you from using, or negatively affect the display or function of, a website or certain areas or features of a website. For more details on cookies, please visit All About Cookies.
Analytics: We may use a variety of online analytics products that use cookies to help us analyze how users use our Services and enhance your experience when you use the Services.
2. How we use personal information
We may use Personal Information for the following purposes:

To provide, administer, maintain and/or analyze the Services;
To improve our Services and conduct research;
To communicate with you; including to send you information about our Services and events;
To develop new programs and services;
To prevent fraud, criminal activity, or misuses of our Services, and to protect the security of our IT systems, architecture, and networks;
To carry out business transfers; and
To comply with legal obligations and legal process and to protect our rights, privacy, safety, or property, and/or that of our affiliates, you, or other third parties.
Aggregated or De-Identified Information. We may aggregate or de-identify Personal Information so that it may no longer be used to identify you and use such information to analyze the effectiveness of our Services, to improve and add features to our Services, to conduct research and for other similar purposes. In addition, from time to time, we may analyze the general behavior and characteristics of users of our Services and share aggregated information like general user statistics with third parties, publish such aggregated information or make such aggregated information generally available. We may collect aggregated information through the Services, through cookies, and through other means described in this Privacy Policy. We will maintain and use de-identified information in anonymous or de-identified form and we will not attempt to reidentify the information, unless required by law.

As noted above, we may use Content you provide us to improve our Services, for example to train the models that power ChatGPT. Read our instructions on how you can opt out of our use of your Content to train our models.

3. Disclosure of personal information
In certain circumstances we may provide your Personal Information to third parties without further notice to you, unless required by the law:

Vendors and Service Providers: To assist us in meeting business operations needs and to perform certain services and functions, we may provide Personal Information to vendors and service providers, including providers of hosting services, customer service vendors, cloud services, email communication software, web analytics services, and other information technology providers, among others. Pursuant to our instructions, these parties will access, process, or store Personal Information only in the course of performing their duties to us.
Business Transfers: If we are involved in strategic transactions, reorganization, bankruptcy, receivership, or transition of service to another provider (collectively, a “Transaction”), your Personal Information and other information may be disclosed in the diligence process with counterparties and others assisting with the Transaction and transferred to a successor or affiliate as part of that Transaction along with other assets.
Legal Requirements: We may share your Personal Information, including information about your interaction with our Services, with government authorities, industry peers, or other third parties (i) if required to do so by law or in the good faith belief that such action is necessary to comply with a legal obligation, (ii) to protect and defend our rights or property, (iii) if we determine, in our sole discretion, that there is a violation of our terms, policies, or the law; (iv) to detect or prevent fraud or other illegal activity; (v) to protect the safety, security, and integrity of our products, employees, or users, or the public, or (vi) to protect against legal liability.
Affiliates: We may disclose Personal Information to our affiliates, meaning an entity that controls, is controlled by, or is under common control with OpenAI. Our affiliates may use the Personal Information we share in a manner consistent with this Privacy Policy.
Business Account Administrators: When you join a ChatGPT Enterprise or business account, the administrators of that account may access and control your OpenAI account. In addition, if you create an account using an email address belonging to your employer or another organization, we may share the fact that you have an OpenAI account and certain account information, such as your email address, with your employer or organization to, for example, enable you to be added to their business account.
Other Users and Third Parties You Share Information With: Certain features allow you to display or share information with other users or third parties. For example, you may share ChatGPT conversations with other users via shared links or send information to third-party applications via custom actions for GPTs. Be sure you trust any user or third party with whom you share information.
4. Your rights
Depending on location, individuals may have certain statutory rights in relation to their Personal Information. For example, you may have the right to:

Access your Personal Information and information relating to how it is processed.
Delete your Personal Information from our records.
Rectify or update your Personal Information.
Transfer your Personal Information to a third party (right to data portability).
Restrict how we process your Personal Information.
Withdraw your consent—where we rely on consent as the legal basis for processing at any time. 
Object to how we process your Personal Information.
Lodge a complaint with your local data protection authority.
You can exercise some of these rights through your OpenAI account. If you are unable to exercise your rights through your account, please submit your request through privacy.openai.com or to dsar@openai.com.

A note about accuracy: Services like ChatGPT generate responses by reading a user’s request and, in response, predicting the words most likely to appear next. In some cases, the words most likely to appear next may not be the most factually accurate. For this reason, you should not rely on the factual accuracy of output from our models. If you notice that ChatGPT output contains factually inaccurate information about you and you would like us to correct the inaccuracy, you may submit a correction request through privacy.openai.com or to dsar@openai.com. Given the technical complexity of how our models work, we may not be able to correct the inaccuracy in every instance. In that case, you may request that we remove your Personal Information from ChatGPT’s output by filling out this form.

For information on how to exercise your rights with respect to data we have collected from the internet to train our models, please see this help center article.

5. Additional U.S. state disclosures
The following table provides additional information about the categories of Personal Information we collect and how we disclose that information. You can read more about the Personal Information we collect in “Personal information we collect” above, how we use Personal Information in “How we use personal information” above, and how we retain Personal Information in “Security and Retention” below.

Category of Personal Information	Disclosure of Personal Information
Identifiers, such as your name, contact details, IP address, and other device identifiers	We may disclose this information to our affiliates, vendors and service providers to process in accordance with our instructions; to law enforcement and other third parties for the legal reasons described above; to parties involved in Transactions; to corporate administrators of enterprise or team accounts; and to other users and third parties you choose to share it with.
Commercial Information, such as your transaction history	We may disclose this information to our affiliates, vendors and service providers to process in accordance with our instructions; to law enforcement and other third parties for the legal reasons described above; to parties involved in Transactions; and to corporate administrators of enterprise or team accounts.
Network Activity Information, such as Content and how you interact with our Services	We may disclose this information to our affiliates, vendors and service providers to process in accordance with our instructions; to law enforcement and other third parties for the legal reasons described above; to parties involved in Transactions; and to other users and third parties you choose to share it with.
Geolocation Data	We may disclose this information to our affiliates, vendors and service providers to process in accordance with our instructions; to law enforcement and other third parties for the legal reasons described above; and to parties involved in Transactions.
Your account login credentials and payment card information (Sensitive Personal Information)	We disclose this information to our affiliates, vendors and service providers, law enforcement, and parties involved in Transactions.
To the extent provided for by local law and subject to applicable exceptions, individuals may have the following privacy rights in relation to their Personal Information:

The right to know information about our processing of your Personal Information, including the specific pieces of Personal Information that we have collected from you;
The right to request deletion of your Personal Information;
The right to correct your Personal Information; and
The right to be free from discrimination relating to the exercise of any of your privacy rights.
We don’t “sell” Personal Information or “share” Personal Information for cross-contextual behavioral advertising (as those terms are defined under applicable local law). We also don’t process sensitive Personal Information for the purposes of inferring characteristics about a consumer.

Exercising Your Rights. To the extent applicable under local law, you can exercise privacy rights described in this section by submitting a request through privacy.openai.com or to dsar@openai.com.

Verification. In order to protect your Personal Information from unauthorized access, change, or deletion, we may require you to verify your credentials before you can submit a request to know, correct, or delete Personal Information. If you do not have an account with us, or if we suspect fraudulent or malicious activity, we may ask you to provide additional Personal Information and proof of residency for verification. If we cannot verify your identity, we will not be able to honor your request.

Authorized Agents. You may also submit a rights request through an authorized agent. If you do so, the agent must present signed written permission to act on your behalf and you may also be required to independently verify your identity and submit proof of your residency with us. Authorized agent requests can be submitted to dsar@openai.com.

Appeals. Depending on where you live, you may have the right to appeal a decision we make relating to requests to exercise your rights under applicable local law. To appeal a decision, please send your request to dsar@openai.com.

6. Children
Our Service is not directed to children under the age of 13. OpenAI does not knowingly collect Personal Information from children under the age of 13. If you have reason to believe that a child under the age of 13 has provided Personal Information to OpenAI through the Service, please email us at legal@openai.com. We will investigate any notification and if appropriate, delete the Personal Information from our systems. If you are 13 or older, but under 18, you must have permission from your parent or guardian to use our Services.

7. Links to other websites
The Service may contain links to other websites not operated or controlled by OpenAI, including social media services (“Third Party Sites”). The information that you share with Third Party Sites will be governed by the specific privacy policies and terms of service of the Third Party Sites and not by this Privacy Policy. By providing these links we do not imply that we endorse or have reviewed these sites. Please contact the Third Party Sites directly for information on their privacy practices and policies.

8. Security and Retention
We implement commercially reasonable technical, administrative, and organizational measures to protect Personal Information both online and offline from loss, misuse, and unauthorized access, disclosure, alteration, or destruction. However, no Internet or email transmission is ever fully secure or error free. In particular, email sent to or from us may not be secure. Therefore, you should take special care in deciding what information you send to us via the Service or email. In addition, we are not responsible for circumvention of any privacy settings or security measures contained on the Service, or third-party websites.

We’ll retain your Personal Information for only as long as we need in order to provide our Service to you, or for other legitimate business purposes such as resolving disputes, safety and security reasons, or complying with our legal obligations. How long we retain Personal Information will depend on a number of factors, such as the amount, nature, and sensitivity of the information, the potential risk of harm from unauthorized use or disclosure, our purpose for processing the information, and any legal requirements.

9. International users
By using our Service, you understand and acknowledge that your Personal Information will be processed and stored in our facilities and servers in the United States and may be disclosed to our service providers and affiliates in other jurisdictions.

Legal Basis for Processing. Our legal bases for processing your Personal Information include:

Performance of a contract with you when we provide and maintain our Services. When we process Account Information, Content, and Technical Information solely to provide our Services to you, this information is necessary to be able to provide our Services. If you do not provide this information, we may not be able to provide our Services to you. 
Our legitimate interests in protecting our Services from abuse, fraud, or security risks, or in developing, improving, or promoting our Services, including when we train our models. This may include the processing of Account Information, Content, Social Information, and Technical Information. Read our instructions on how you can opt out of our use of your information to train our models.
Your consent when we ask for your consent to process your Personal Information for a specific purpose that we communicate to you. You have the right to withdraw your consent at any time.
Compliance with our legal obligations when we use your Personal Information to comply with applicable law or when we protect our or our affiliates’, users’, or third parties’ rights, safety, and property.
Data Transfers. Where required, we will use appropriate safeguards for transferring Personal Information outside of certain countries. We will only transfer Personal Information pursuant to a legally valid transfer mechanism.

Data Protection Officer. You can contact our data protection officer at privacy@openai.com in matters related to Personal Information processing.

10. Changes to the privacy policy
We may update this Privacy Policy from time to time. When we do, we will post an updated version on this page, unless another type of notice is required by applicable law.
**Post 3:**

> Privacy impact of Automatic License Plate Readers (ALPRs) >"Those who would give up essential Liberty, to purchase a little temporary Safety, deserve neither Liberty nor Safety” - Benjamin Franklin

Forgive the long post, but this is an important topic that has not gotten nearly enough attention. My HOA is considering installing Automatic License Plate Readers (ALPRs) throughout my neighborhood. In my research, I've become more and more concerned about the privacy impact. This writeup has a focus on Washington State, but this is applicable throughout the US. Laws and impacts may vary in other countries.

# Dangers

Let's look deeper into how law enforcement uses Automatic License Plate Readers (ALPR) as investigated by reporters and researchers. First, some real examples of this tracking technology endangering innocent lives.

[Based on ALPR data, police mistakenly hauled children out of a car at gunpoint and handcuffed them.](https://www.9news.com/article/news/local/aurora-police-chief-says-she-is-deeply-troubled-after-her-officers-detained-and-handcuffed-kids-while-mistaking-a-car-for-stolen/73-bb653e17-a208-4f21-a972-c5d0f0889bfc)

[Based on ALPR data, an innocent man and his passenger were held at gunpoint for driving a rental car that was mistakenly listed on the hot list.](https://www.ktvu.com/news/privacy-advocate-sues-coco-sheriffs-deputies-after-license-plate-readers-target-his-car-stolen)

[Based on ALPR data, a police lieutenant illegally tracked the location of his estranged wife.](https://www.kwch.com/2022/10/31/kechi-police-lieutenant-arrested-using-police-technology-stalk-wife/)

[ICE and CBP are using is using license plate data harvested by ALPRs to arrest people at their homes and in their community without a warrant.](https://www.aclu.org/news/immigrants-rights/documents-reveal-ice-using-driver-location-data) We already know by reading today's news that the current government has silently revoked visas, and deported foreign students as well as actual US citizens and children.

By contracting out to unregulated private companies to perform key elements of its investigative work, ICE sidesteps the need for search warrants or other forms of collaboration by local governments in the jurisdictions in which it operates. This not only streamlines their process, but essentially removes their activities from the oversight of courts in our communities. Further, [there is reporting that Spokane shares its data with agencies in Idaho](https://jsis.washington.edu/humanrights/2022/12/07/whos-watching-washington/). Idaho has a "bounty hunter" law that rewards anyone who reports someone traveling to Spokane (or elsewhere in WA) for an abortion/reproductive health care that is illegal in Idaho.

# Protections?

Those are some of the many potential dangers of automated tracking technology and mass surveillance. So what protections are in place?

In Washington, there are *no* state laws regulating the use of ALPRs. A bill was proposed to put safeguards in place, but it was not passed. HB 1909 would have restricted ALPRs in the following way.

>If the image or data does not match a license plate number on the watch list, the image or data must not be: Used to identify the owner or driver of a vehicle; shared with any other agency, entity, or person; used for any other purpose; or retained for more than twelve hours.

It is notable that retention of data about innocent vehicles was to be limited to 12 hours, whereas tracking companies like Flock want to retain it for at least 30 days. Laws regulating ALPRs exist in about 15 other states, all with different levels of protections involved. Many more have pending legislation, but no guarantee they will pass. It is reasonable to assume that the WA state legislature will pick up the topic again, *if* there is a push from citizens.

Washington State police agencies do have [a set of guidelines](https://waautotheftpreventionauthority.org/wp-content/uploads/2022/09/WA-License-Plate-Reader-Guidelines.pdf) in place, however they state

>The Automated License Plate Reader Guidelines are non-binding guidelines, voluntarily adopted by the Washington Association of Sheriffs and Police Chiefs.

This means they can be changed at any time, and there are no ramifications for following them or not. In fact, some agencies, including Fife and Edmonds, already do not perform the annual audits that the guidelines suggest. Therefore there is no way to know if agencies are complying with any of the other guidelines. In Washington state, many police agencies don't even know who they are sharing the data with or even how to find out.

Audits are extremely important. An audit, required by law in California, revealed that "due to confusing settings" [3 different ICE agencies had access to ALPR data, despite their efforts to disallow that access](https://lawshun.com/article/can-anyone-besides-law-enforcement-use-license-plate-scanners). Without the required audit, this would never have been caught.

# Privacy

According to legal experts, the more ALPRs that are interconnected, the more likely it will be determined to violate a person's right to privacy. **Yes, even when traveling through public spaces.** Because of this, even those in favor of this technology may be better served to limit the rollout of these trackers. In Commonwealth vs McCarthy, a judge ruled:

>if the State police had obtained historical locational data regarding the defendant's vehicle from enough automatic license plate readers (ALRPs) in enough locations, the mosaic that such collection would create of the defendant's movements 'would invade a reasonable expectation of privacy and would constitute a search for constitutional purposes.'

[Lawyer Steve Graham further explains](https://www.grahamlawyerblog.com/2024/09/03/law-enforcement-use-of-license-plate-readers-in-washington-state/):

>Several courts have cautioned, however, that if the ubiquity of ALPR ever is such that a person’s tracking is continuous, then that database should not be accessed by law enforcement on whim. Rather the police would need to have probable cause and likely need a warrant. In this sense, ALPR risks being a victim of its own success. The more data the companies collect, the greater the likelihood that law enforcement could no longer routinely access the information. If the police wish to track someone based on their cell phone location, they need probable cause and a warrant to access these records from a phone company.

If any of this concerns you with regard to HOA-operated ALPRs, keep in mind that you do not have Fourth Amendment rights with respect to private entities like HOAs, private security firms, etc. The Constitution applies only to government agencies and law enforcement. LE is pushing private entities to fund and roll out tracking devices because it allows LE to skirt the Fourth Amendment. Private entities are also not subject to public information requests, and so no accountability or auditing is done to ensure compliance with any laws or regulations.

Let's not forget that Flock, a private company with no oversight, [asserts the right that all footage and data can be shared with any third party by them at any time, if they believe it is in the interest of public safety](https://www.flocksafety.com/lpr-policy). Their whole business operates under the assumption that every image and datapoint is in the name of public safety. A simple request by any government agency will trivially meet that criteria, as would any financial deal they make to sell data to other "public safety" companies.

# Civil Liberties Groups

ALPRS are **opposed** by the [Electronics Frontier Foundation (EFF)](https://www.eff.org/deeplinks/2020/09/flock-license-plate-reader-homeowners-association-safe-problems), a nonprofit supporting civil liberties in the digital world.

ALPRs are **opposed** by the [Americal Civil Liberties Union (ACLU).](https://www.aclum.org/en/publications/what-you-need-know-about-automatic-license-plate-readers)

ALPRs are **opposed** by the [University of Washington Center for Human Rights (UWCHR).](https://jsis.washington.edu/humanrights/2022/12/07/whos-watching-washington/)

I encourage you to read through all of the above for explanations, but their main reasons are:

* No actual evidence that ALPRs reduce crime
* Lack of oversight
* Misreads and false hits
* Misuse by Law Enforcement Officers (LEO) and others with access to the database

If your state doesn't already have protections in place, please contact your state representatives and ask, beg, or demand them to bring up legislation as soon as possible. Only together can we slow or block the mass surveillance from tracking our every move.
**Post 4:**

> TSA again backs down from it's REAL-ID threats again  [https://papersplease.org/wp/2024/09/16/tsa-again-backs-down-from-its-real-id-threats/](https://papersplease.org/wp/2024/09/16/tsa-again-backs-down-from-its-real-id-threats/)

The Transportation Security Administration (TSA) has again backed down from its decades-old threats to start requiring all airline passengers to show ID that the TSA deems to be compliant with the [REAL-ID Act of 2004.](https://papersplease.org/wp/category/real-id/) But the new rules proposed by the TSA would create new problems that won’t go away until Congress repeals the REAL-ID Act.

In a [notice published in the Federal Register on September 12th](https://www.regulations.gov/document/TSA-2023-0003-0001) , the TSA has proposed another two-year postponement of the [most recent  of the “deadlines” the agency has imposed on itself](https://papersplease.org/wp/2022/12/05/dhs-resets-the-clock-on-its-threat-to-stop-flyers-without-id/) for REAL-ID enforcement.  But that postponement would be combined  with [interim rules for the next two years](https://www.regulations.gov/document/TSA-2023-0003-0001) that ignore the law and invite arbitrariness in how travelers are treated.

The TSA notes that “frustrated travelers at the checkpoint may also increase security risks” if the TSA stopped allowing travelers to fly without REAL-ID. But the TSA doesn’t mention its current [procedures for flying without any ID](https://papersplease.org/wp/2021/04/08/tsa-posts-video-showing-how-you-can-fly-without-id/) or its[ position in litigation](https://papersplease.org/wp/2023/06/12/tsa-misstates-the-case-law-on-id-to-fly/) that no law or regulation requires airline passengers to show *any* ID. Instead, The TSA claims without explanation that without this postponment, “individuals without  REAL ID-compliant DL/ID or acceptable alternative would be unable to board federally regulated aircraft.”

[Comments from the public](https://www.regulations.gov/docket/TSA-2023-0003/document) on the proposed rule are due by October 15, 2024. [Dozens of comments](https://www.regulations.gov/docket/TSA-2023-0003/comments) have already been submitted, almost all of them opposing requiring REAL-ID to fly.

We’ll be submitting comments opposing the proposed rules and reminding the TSA that (1) [no state is yet in compliance with the REAL-ID Act](https://papersplease.org/wp/wp-content/uploads/2023/10/IDP-mDL-comments-16OCT2023.pdf), which would require sharing of driver and ID databases with *all* other states, and (2) [neither the REAL-ID Act nor any other Federal law requires air travelers to have, to carry, or to show any ID](https://papersplease.org/wp/2024/05/22/new-dhs-publicity-about-real-id/).

Unless the law is changed to try to impose an unconstitutional ID requirement as a condition on the right to travel by common carrier, the TSA must [continue to recognize the right to fly without ID](https://papersplease.org/wp/2017/04/13/yes-you-can-fly-without-id/). Any distinction by the TSA or other Federal agencies between state-issued ID, when no state complies with the REAL-ID Act or could do so until all states participate in the [national REAL-ID database](https://papersplease.org/wp/2016/02/11/how-the-real-id-act-is-creating-a-national-id-database/) (SPEXS), would be arbitrary and unlawful.

The TSA is [proposing](https://www.regulations.gov/document/TSA-2023-0003-0001) what it describes as phased enforcement of the prohibition on acceptance by Federal agencies (in circumstances, which don’t include travel by common carrier, in which the law requires individuals to show ID) of ID that doesn’t comply with the REAL-ID Act.

Under the rules proposed by the TSA, full enforcement of the REAL-ID Act would be postponed by another two years, from the previous arbitrary deadline of May 7, 2025, to a new and equally arbitrary deadline (subject to further postponments) of May 5, 2027.

In the interim, the rules proposed by the TSA would authorize the TSA itself and all other Federal agencies to engage in graduated enforcement measures against individuals who don’t have ID the Department of Homeland Security (DHS) deems to be compliant with the REAL-ID Act

The TSA suggests that graduated enforcement measures might include the creation of new databases tracking the use of “noncompliant” ID, on the basis of which agencies would limit the number of times an individual could enter a Federal building or engage in other activities without  showing “compliant” ID. But the TSA woud leave it to agencies’ discretion to decide what conditions to impose on use of noncompliant ID.

These graduated enforcement policies would determine who is, and who is not, able to exercise Federally-protected rights. Contrary to the Administrative Procedure Act (APA), those policies would not be specified in regulations published in the Federal Register. They would be adopted by [individual agencies](https://papersplease.org/wp/2020/11/10/what-does-real-id-act-enforcement-really-mean/) without prior notice or comment.

Even if the REAL-ID Act had authorized the TSA to delegate to other Federal departments authority to issue agency-specific REAL-ID enforcement rules, which it didn’t, those regulations would be subject to the APA. The TSA [asked Congress to exempt REAL-ID Act implementation](https://papersplease.org/wp/2020/07/28/senate-bill-would-exempt-real-id-from-due-process-and-oversight/) from the APA and the Paperwork Reduction Act (PRA), but Congress [declined to enact those exemptions](https://papersplease.org/wp/2021/01/03/real-id-act-amended-but-dhs-doesnt-get-the-exemptions-it-wanted/).

In its latest [Notice of Proposed Rulememaking](https://www.regulations.gov/document/TSA-2023-0003-0001) (NPRM), the TSA acknowledges that the PRA and the Privacy Act  would require agencies to publish Federal Register notices and obtain approval by the Office of Management and Budget (OMB) for new databases and collection of information about users of noncompliant ID. But the  TSA doesn’t mention the APA. It acts as though Congress approved the APA exemption it rejected for REAL-ID rules.

The TSA’s latest notice doesn’t mention the agency’s [most recent previous proposal](https://papersplease.org/wp/2023/10/16/the-tsa-wants-to-put-a-government-tracking-app-on-your-smartphone/) related to REAL-ID, which would have authorized the use of smartphone traveler-tracking apps as an alternative to REAL-ID drivers license or ID cards. That proposal hasn’t been finalized, but [additional background information](https://www.regulations.gov/docket/TSA-2023-0002/document?sortBy=postedDate) was disclosed last month, perhaps indicating preparations by the TSA for a new round of comments on that proposed rule.
**Post 5:**

> One way age verification could be exploited by criminals I believe it is crucial to bring attention to the potential risks of age verification methods, particularly in obtaining recordings of individuals without their explicit consent. I seek to promote how actively harmful these policies are, with specific concern to the privacy and security of women.

This post will explore the potential risks associated with fraudulent age verification websites, specifically addressing their capacity to conduct unauthorized surveillance on users by webcams. This post discusses age verification methods reliant on webcams, including those involving document and biometric verification techniques.

I hope the information is clear, I am considering further elaboration on this topic in a potential future article.

One method threat actors can use is the manipulation of the Domain Name System (DNS), a system responsible for translating domain names into corresponding IP addresses. 

DNS hijacking involves a threat actor altering DNS settings to redirect users to undesired content, typically ads or an alternative search engines. This same approach is also utilized for DNS blocking, as seen in the EU, where Internet Service Providers employ it to restrict access to pirate websites, due to its efficiency and minimal collateral damage (compared to IP blocking which could block totally innocent sites).

I personally encountered a situation earlier this year where my Google Chrome browser repeatedly redirected to Bing and displayed ads for blatant drop-shipping sites. Upon investigation, I discovered that a video download extension I had installed was the source of the issue.

Whether it was inherently malicious or became compromised through acquisition or updates is anybody’s guess, however the problems ceased upon removal of the extension. I was not aware of the situation primarily because of my consistent use of a virtual privacy network, which modifies my DNS settings and likely prevented the malicious extension from redirecting me from Google to Bing or serving ads. I am sure it was a DNS hijacker and not adware due to that. 

Also In 2022, it was discovered that a browser colour modification extension exhibited similar behaviour, these extensions are out there and can be abused.

It is conceivable for threat actors to use this method, redirecting individuals attempting to access adult websites to deceptive or otherwise bogus age verification website. A threat actor could discreetly establish an innocent looking extension, purporting to provide something like local weather information or enhance website font size on websites for people with eye sight issues. Then when a user attempts to access an adult website, the extension could redirect them to a fake age verification site. Something like a fake YOTi page. 

This deceptive age verification website may coerce users into granting access to their webcam under the guise of biometric or document based age verification, enabling the unconsensual recording of the user's webcam activity, It’s a creeps dream. 

The risk of such crimes does not necessitate any downloads as well; rather, threat actors could easily establish a deceptive adult website hidden behind age verification measures. 

There is already a fake Pornhub website that exists with an obscure domain extension, it’s obviously a fake Pornhub site when you visit it. The primary purpose of this site is not data theft or deception; rather, it serves to circumvent restrictions present in certain countries such as India and Pakistan. To bypass blocks on adult websites in these regions, the content is hosted on another separate domain. 

However, obscured by an age verification page, the inherent risks associated with this website significantly increase. The website's legitimacy is unable to be verified without potentially giving it your sensitive info and allowing it access to your webcam. This situation even enables the potential for threat actors to populate a site with illegal material, waiting until a victim verifies their age and identity to then exploit them through blackmail. People won’t go to the police, people will take their lives. 

I have yet to address additional potential risks associated with these systems, such as invasive webcam surveillance akin to how Remote Administration Tools (RATS) are used now, the majority of victims being women or teenagers. 

The lobbying efforts undertaken by age verification companies in support of these policies have, in certain instances, positioned them in a manner that potentially shields them from legal or regulatory repercussions by the government.. For example, within the context of the United Kingdom, the Age Verification Providers Association, a profit oriented trade group actively involved in lobbying the government for age verification implementation, could potentially exploit their ability to withdraw from the UK jurisdiction in response to any intervention by regulatory bodies such as the Information Commissioner's Office (ICO) or the Department for Digital, Culture, Media & Sport (DCMS).

This situation presents a catch 22 moment for the government, creating a situation where pursuing legal action against these entities becomes a challenging prospect. The fear of backlash from various "children safety" advocacy groups and campaigners could cast the government as weak towards adult websites. 

An instance highlighting this issue arose when it was disclosed that the Department of Education had shared children's information with betting firms to prevent the opening of accounts by underage individuals. Given the timeframe, it is likely that my data was included in this.

Rather than pursuing formal action against the Department of Education, the Information Commissioner's Office (ICO) sent their concerns through a strongly worded letter. Despite the potential risks, such as the misplacement of this data or a threat actor carrying out a data breach once they reach adulthood.

A similar response may be expected in the event that age verification companies like YOTi are implicated in recording users and selling such information to data brokers—though it’s crucial to clarify that I am not asserting YOTi’s involvement in such a scheme. However, the implementation of policies involving a severe intrusion into people’s privacy increases the likelihood of such occurrences over time. 

Profit over privacy (and people)…
## Topic 1: Parenting & Youth Challenges
**Post 1:**

> Deep Dive into the SMM Panel Industry’s Secrets from a 10-Year SMM Panel Owner – Best & Cheapest SMM Panel? Oldest SMM Panels? First SMM Panel & Biggest SMM Panel | ASK ME ANYTHING Hello, 

I am Dmytro Chumak from Ukraine. If I were to briefly describe myself, it would be most accurate to say that I have spent the last 10 years as both an SMM panel owner, user, and reseller. For 8 years, I managed my own automated Smm Panel software, and I might be the first person of Ukrainian origin to operate such a system. During those 8 years, I worked with the largest smm panel provider + I sold my own data services to today's top SMM panels when they had only 5,000-10,000 orders. In total, I made sales close to 10 million, earned substantial amounts of money, and, as an inevitable part of the SMM panel sector, got scammed by fraudulent panels. My friends include the owners of Justanotherpanel, Morethanpanel, Bulkfollows, and Smmraja, to whom I sold and from whom I bought services. I was the 5th SMM panel to start using the PerfectPanel system and have a long-standing friendship with Victor (the owner of PerfectPanel).  

Now, why am I telling you all this? As you know, the SMM panel sector is a boundless abyss. Unlike when we started, there are now thousands of social media reseller panels, and unfortunately, 99% of them are scammers. In this tangled web of providers and resellers, I wanted to share all my experiences and the secrets of the SMM panel world (which nobody wants to share on the internet) with you, having spent 10 years in this sector and exited about 2 years ago. 

My aim is to create a very clean roadmap for those who want to take their first step in this sector. I hope this article helps you. Please do not hesitate to ask me any quesion

\-------------------------------------------------------------------------------

# ✅ Is There a Single Main Provider Panel? If so Which Ones are They? 

This is probably the most frequently asked question about the SMM panel industry :) No, friends, there isn't a single main SMM service provider panel. Quite the opposite, there are providers just in their own field or social media niche markets. In short, you can never have all real provider services through a single panel. Another common misconception is that the main provider always offers the cheapest price and stable services. No, friends! Even if you find a main provider panel, there is a 70-80% chance that the services of these provider panels are cheaper on today's largest SMM panels. So

🧠 LESSON 1- Do not focus on the find provider, focus on the advance support and **high completion rate panels** :)   

\-----------------------------------------------------------------------  


# ✅ How can We Understand If a Panel is a Main Provider or Not? 

Very clearly! Main provider SMM panels focus only on specific social media channels. The claim that a panel, claiming to be a main provider, provides different services for 4-5 platforms is completely false. Although I cannot vouch for any panel, the best example in the sector is StreamingMafia (formerly Spotifypanel). They have a single focus: Spotify services, and after using the panel a bit, you can easily tell that 70-80% of the data is their own. 

The second method of understanding is through prices. You will never see a fractionated price on a true SMM provider panel. You will not see a price like $10.123 on any legit provider panel. This is a reseller SMM panel price that includes their commission. Real providers use certain price values like $1 - $2.50 

\-----------------------------------------------------------------------------

# ✅ How Do Main Providers Offer These Services? What Kind of Bot System Do They Use? 

Ten years ago, everything was so simple that I can even say without hesitation that I was the first person in the sector to have a Facebook follower service (even though I had no bot knowledge). With minimal automation and coding knowledge, you could create your own social media bot system, and it would work stably. However, with the ever-growing digital world and social media platforms taking measures against bots, this job now requires deep coding knowledge.

Let me immediately say that if you think you can download a bot from GitHub or buy a social media bot sold on a forum and enter this sector as a provider, forget it. Social media platforms release updates almost every week. The key is not just to get or find a working bot, but to ensure that the bot you purchase is regularly updated every week. This means that at least one person on your team must have advanced coding knowledge. 

If you ask what the most commonly used bot coding system is, the answer is "REQUEST". This is the system used by 95% of real SMM service providers worldwide (those with some coding knowledge will understand what I mean).  

# There are thousands of different SMM panel resellers and providers from India, Bangladesh, Turkey, Russia, and the USA. If We are Using an SMM Panel for the First Time, Should We Consider the Region of the Panel? 

Both yes and no. First, you need to know that the backbone of this global SMM industry is formed by Indians and Bangladeshis. There are both very large providers and the biggest scammers among them. Generally, you can find cheap services in these countries, but the cheaper the prices, the lower the order completion rate and customer support (e.g., peakerr - Elitesmm). 

Turkish panels rank 3rd. They are mid-tier SMM providers. They are neither as cheap as Indian panels nor as bad in support (e.g., Smmturk - Bestsmmprovider). 

Russian and Ukrainian panels are more focused on the back-end provider side. Their support is very poor, but their bots are as good as those of the Indians. 

European and American panels: Although they entered the sector 3-4 years later than other regions, I can comfortably say that these countries’ panels are currently the best in terms of customer satisfaction, secure payment gateways, and service uptime (e.g., Morethanpanel - Fivebbc - Clousty). 

  
\-----------------------------------------------------------------------------

# ✅ I Want to Open an SMM Panel. Do You Think It is Still Worth Investing in This Business, or More Explicitly, Can I Make Money? 

Both yes and no. If your business plan is to open a generic SMM reseller panel and sell by adding a 25-30% profit margin to your provider prices, I would say don't start this business at all. It is almost impossible to gain volume or make a profit by doing reseller sales in a market dominated by giants like Justanotherpanel, Morethanpanel, and Bulkfollows. Unfortunately, those who opened basic smm panels this way between 2015-2019 all made a fortune, but in 2024 circumstances, this is nearly impossible. 

So how can we make money as smm panel owner? 

You need to specialize in a SMM panel niche. Your goal should not be to sell all services but to sell just one service. If it’s YouTube, you should be solely a YouTube panel. You need to find the biggest providers in the YouTube space and establish your SMM panel with excellent customer support. 

You need to have your own custom-built bot system :) This is the shortest and most direct way, friends. With a very basic TikTok follower bot, there is nothing stopping you from reaching daily sales of $3-5k within a month. 

  
\-------------------------------------------------------------------------

# ✅ What are the Oldest SMM Panels and Which is the Largest in Terms of Sales volume? 



Unfortunately, the first or oldest SMM panels are no longer active websites. The first SMM panels were likely established around 2013 when Instagram became popular, but they were not exactly like today's automated SMM panels. The foundations of today's systems were laid around 2016, but we can safely say that sites like igfollowers.com, igrbooster.net, and youtubelikes.com were the ancestors of the first SMM panels. 

Among the SMM panels still active today, Morethanpanel former bestpanel), Peakkerr, and Smmfollows are easily the oldest. 

In 2025, we can easily point to Justanotherpanel and Morethanpanel as the biggest SMM panel by order volume.  

\--------------------------------------------------------------------------

# ✅ For Those Wondering, "Hey, weren't you going to tell us the best SMM panel providers or the best SMM panels you've used?" This Section is For You :)

Yes, friends, I have tried to share my 10 years of experience in this article, but I know that ultimately everyone is curious about one thing: What are, in my opinion, the best SMM panels and top-rated SMM panel providers?

Unfortunately, most of the real SMM providers from 5-6 years ago are no longer selling today (I'm sure most have found much easier ways to make money like I did). To avoid misleading anyone or promoting any particular panel, I would like to share some panels that I know are providers for each social media platform at the beginner level (let me reiterate that no provider always delivers 100%, so keep this in mind when placing your orders).

* **Instagram Provider:** Morethanpanel - Igvoe - Postlikes 
* **YouTube Provider:** Tube-boost
* **Spotify - Music Services:** StreamingMafia ( former spotifypanel ) 
* **Reddit Provider:** Boostupsmm 
* **Facebook Provider:** fastsmm 
* **Twitch Provider:** Globalsmm - StreamPromotion
* **TikTok Provider:** Morethanpanel

(psi: we almost know all providers on the market but I don’t wanna brand this thread as some promotional SMM list – above ones are the most stable providers for long years that I can easily reference.)

Let me reiterate that these are the panels we have used for long years and I’m sure that they are providers, but never forget that working directly with **real SMM panel providers** is not always the best solution. It might even be more expensive than a reseller, or the **reseller SMM panel’s support team** might make you happier than the provider’s support. Because they have higher volumes, they have more leverage over the providers.

# ✅ Finally, let’s address the last topic: What are the Best SMM Panels We Have Used So Far, or More Precisely, Which SMM Panels Would I Use if I Started from Scratch Today as resellers? (a mix of providers and resellers).

Actually, this should have been the first question SMM panel users or beginners need to ask. Unfortunately, everyone looks for providers and ends up getting scammed by the cheapest SMM panels. Our main goal should be to find panels that are as affordable, safe, and well-supported as possible. Based on our 10 years of experience, the panels that have almost never given us any problems whenever we had idle services or used their services are:

**Morethanpanel** – Formerly Bestpanel  
Morethanpanel.com. A site that sells both its own bots (TikTok – Twitch – Spotify, as far as I know) and services with very low reseller margins. Nearly 25% of our services were connected to this panel until 4–5 months before we put our panel up for sale, and I can say their support speed is by far the best on the internet. A great example of a high-performance SMM platform.

**Smmraja** – One of the oldest but one of the best. Prices are a bit higher than Morethanpanel, but the service variety is almost five times more. Great option for people looking to run a white-label SMM reseller panel with diverse services.

Fivebbc – If you want to sell guaranteed high-quality services despite the high cost and your customers are exclusive, this should be your go-to panel. Their service prices are quite high, but that’s exactly the niche they occupy—expensive but premium SMM services.

I’m sure there are cheaper panels than those ones, but as I underline in this thread 3x times: Your main priority needs to be finding the most stable SMM panel with solid customer support and low service drop rates.

  
\-------------------------------------------------------------------------------------

# ✅ For Those Who Already Have an SMM Panel but Want to Know the Best Marketing Strategies for SMM Panel:

Up to this point, I've talked to you as much as possible about how to find the right panels and providers, friends. But unfortunately, even if you are 100% successful in this matter, it doesn't matter at all unless you can drive traffic to your SMM panel website. So, the answer to how we will drive traffic to our panel lies in SMM panel marketing strategies. Here are the most effective marketing methods we have observed so far, friends:

1. Of course, Blackhatworld: I can say openly that this is the first place a person entering the SMM panel business would visit. It's an incredible source of traffic. If you are opening an SMM panel for the first time, you definitely need to have a thread in the Blackhatworld SMM section.
2. SEO: Yes, it's the hardest but the most effective method that will bring you millions if you succeed. Because thousands of people still have their first cheap SMM panel experience with Google searches like "buy SMM panel" or "SMM reseller panel."
3. Google Ads: As you know, Google is quite problematic when it comes to advertising SMM panels. If you try to do this on your own, there is a 99% chance that your Google Ads account will be banned. The best method is to work with an advertising manager who specializes in advertising in such sectors.
4. SMM Listing Websites: If it's free, add your panel to all SMM panel listing directories on the internet without wasting any time.
5. Quora – Reddit: I don't think Quora is as strong as it used to be, but Reddit is still a very powerful source for organic traffic to your SMM panel. Just be careful not to advertise your panel openly, or you'll get banned immediately.
6. Discord Group: Especially effective for India and Pakistan. Either announce your panel in relevant Discord channels or advertise directly on your own server. These groups are goldmines for niche SMM buyers.
7. Skype – Telegram Spam: Yes, most of you might be hearing this for the first time, but the SMM world, especially SMM providers and big resellers, operates entirely through Skype and Telegram. There are hundreds of groups on Skype where SMM panel owners gather. Join these groups and promote your services.

**So that’s all from me, mates. Hope you find these threads useful and help you during your SMM panel journey.**

Please do not hesitate to ask your questions under this thread.

XXX  
**Dmytro Chumak**
**Post 2:**

> Here’s what happened in marketing & social recently 
#Top 6 Updates of the Week: 

* JP Morgan Chase launches digital media business, retail media continues to win.  
* TikTok confirms they are working on a IG Competitor, TikTok Notes for photos content. 
* Meta is testing an AI-powered search bar in Instagram.  
* AirChat is the new trending social media platform, it’s audio only. 
* Youtube threatening TikTok with new shopping & affiliate features. 
* WPP & Google Cloud partner to integrate Gemini in WPP Open Marketing OS.  
* Questionable: France’s PM says, Social Media is a threat to the French language. 

#Trending

* Snapchat expected to earn $65 million in political spend in 2024.  
* LinkedIn pushes collaborative articles further with new  ‘contribute expertise’ shortcut.   
* Paramount Global and Skydance Media are planning a merger. 
* 5 Workplace feature launches from Google Cloud Next’24.  
* Chase Bank will let Advertisers target customers based on spending data. 
* Publicis Groups posts 5.3% growth in Q1 2024. 
* eBay launches global media account review.   
* Diet Coke continues to push further on their social proof strategy in latest ad campaign. 
* Consumers will finally see FCC-mandated nutrition labels for most broadband plans,  
* Puma launches its first worldwide brand campaign in last 10 years: FOREVER FASTER.  
* Discord finally launches chat polls.
* Lucozade goes through first major brand redesign in its 97-year long existence.  

#TikTok 🎶 
* TikTok testing virtual influencers for video ads.  
* TikTok adds new brand safety features: category exclusion and vertical sensitivity controls. 
* TikTok Lite, a new app that rewards users for their screen time.  
* Bytedance overtook Tencent with $40bn profit in 2023.
* TikTok publishes new guide on using ‘promote’ quick advertising option. 
* Televisaunivision plans to bring TikTok-style content to TV with new ad product.  
* Taylor Swift songs return to TikTok. 
* You can now upload 30-minute long videos on TikTok. 

Instagram & Threads 🗂️ 

* Meta launches new tools to protect young teens and users from online exploitation. 
* Threads releases API documentation.   
* Instagram generates more ad revenue than YouTube. 
* Instagram working on ‘Visibility off IG’ setting allowing users to control where their content shows up on internet.   
* Instagram launches ‘IG University’ account and broadcast channel. 
* IG makes story upload faster with a new shortcut to add a new story after a new upload. 
* Facebook testing a new sponsored and product-related content feed. 
* Threads testing a ‘Live Trending’ feature to highlight ongoing discussions. 
* The Meta AI writing feature coming to Stories, allowing you to rewrite story text. 

#Meta 😅


* Meta adds new chat features: HD Photos, Shared Albums & more on messenger. 
* Meta advertisers report Automated System has been glitchy for weeks. 
* Meta announces new Developer platform resources. 
* Meta planing to launch small versions of Llama 3 next week. 
* Meta shares new features and collaborations to build cricket experience in India. 
* WhatsApp is planning to launch an updated image editing experience. 
* WhatsApp’s Meta AI chat feature is now live for more users. 


#X (Twitter) 🕹️

* X removes the ability to hide blue checkmarks. 
* X enables Passkey Access for all iOS users worldwide. 
* Elon Musk announces new algorithm update is coming with bangers and less clickbait. 
* Community Notes will now show notes 30 minutes faster than before.  
* X will soon let you add code blocks to articles. 
* You will soon be able to search for users on X spaces and send invites.   
* X’s Gork AI can now understand images. 
* X automatically changed ‘Twitter’ to ‘X’ in user posts causing links to break. 


#Youtube 🕹️

* Youtube’s new video player layout on desktop is causing huge backlash from users. 
* Youtube switching to read-only comments for kids in supervised experience. 


#Google 🔦 
 
* Google limits News Links in California over proposed ‘Link Tax’ law. 
* Google updates Merchant Center product data specifications. 
* Google updates Carousels Structured Data Documentation. 
* Google improves INP for websites using consent management platforms. 
* Google testing a search tab for short form videos. 
* Google contract shows deal with Israel Defence Ministry.  
* Google confirms major Gmail AI security update for all users.  
* Google to shut down its One VPN Service. 

 
#Agency News 

* Havas acquires TED Consulting to level up AI and data capabilities. 
* Wellness pet Company hires Colle McVoy as Creative AOR. 
* Specsavers appoints Impression Digital as new Digital marketing partner. 
* Pit Viper appoints DEPT ® as Paid Media Agency of Record. 
* Red Brick Road Agency wins Skipton Building Society account. 
* M&C Saatchi steals McCann UK’s creative leaders; Rob Doubal and Laurence Thomson.  
* London-Based Marketing Consultancy Rival acquires creative shop HeyLet’sGo. 
* Kathmandu appoints Motion Sickness as Creative AOR. 
* GroupM Government Unit selected as Media AOR for Gov Canada. 
* Honda powersports appoints Cairns Oneil as Media Agency of Record. 
* Mother Design hires Ariadna Navarro as its First President. 
* Volkswagen partners with Team ITG on comms activation.  
* Insomnia Cookies hires Earnies as new comms agency. 
* The Cotswold company hires the7stars to handle creative and media accounts. 
* BAFTA appoints We Are Social Studio as the Film and Television Awards. 

#Brands & Ads 🏓 

My Fav; Marmite’s new ad campaign aims to hire smugglers to solve New York City’s marmite scarcity problem. 

* The Lego Group is rolling out a new design identity in collaboration with Interbrand consultancy. 
* Powerade’s global campaign ahead of Summer Olympics focuses on Athlete’s mental and physical health.   
* Ford and VML launch new campaign in NZ celebrating success of Ford’s Ranger series. 
* Oreo’s new commerce activation uses website hamburger menu designs to handout discounts to consumers. 
* Heinz’s latest campaign highlights even food stylists can’t fake Heinz ketchup, nothing fake in Heinz from Reel to Real. 
* Dove pledges to never use AI in advertising, comparing real beauty to AI imagination in new campaign. 
* Hennessy refreshes brand with new ‘Made for More’ Campaign and activations at Dreamville festival.  
* Canal+ and BETC Paris launch a new campaign to promote short film offerings.  
* Heineken’s new global campaign uses ‘The First Ahh’ as a message to connect a sense of celebration with the product arrival. 
* L’Oréal Paris is putting celebrity resumes on LinkedIn, celebrating setbacks.  
* Kia uses storytelling abilities and influence of Rafael Nadal in the latest ad campaign.  
* Pedigree promotes love at first sight, but replace humans with adorable dogs.  
* Vienna’s Tourist Board launches Digestible Micro-Doses in latest campaign promoting vibes of the city. 
* Burger King’s new ‘foodfillment’ campaign uses the sense of satisfaction we all have after the last bite. 
* The White Claw’s effort to elevate Social Connections between Gen-Z adults falls short in the latest campaign.   
* Sprite’s very unique summer campaign, launching ‘The Knocklace collection’ for knock-knock. 


#AI 🤨

* Google & MIT RAISE launch a new generative course for educators. 
* GPT-4 Turbo is now available to paid GPT subscribers. 
* Google’s new AI campaign highlights how AI helps to restore Tassie’s Giant Kelp Forests. 
* Pedigree uses AI in latest ad campaign to level up Dog cuteness.   
* SAP Emarsys launches AI Product finder. 
* Orange AI expands their Google Cloud partnership. 
* Adobe is out on the streets buying video data for AI training at the price of $3/minute. 

#Snapchat & Pinterest


* Snapchat partners with Snowflake to build multiple-party clean rooms. 
* Megan Thee Stallion shares Pinterest is her favourite platform. 
* Snapchat will return to the IAB NewFronts with new AdTech announcements. 

#Microsoft & LinkedIn

* LinkedIn shares a new tip with a claim: commenting once a week can triple your profile views.  
* LinkedIn sharing more information about influencers who view your profile. 
* Microsoft starts testing ads in the Windows 11 Start menu. 
* LinkedIn is rolling out a verification badge for recruiters.  

#Marketing & AdTech 

* Sincera’s new dashboard to track Privacy Sandbox API adoption. 
* JCDecaux Australia launches broadcast-scale 3D nationwide. 
* Bark Air launches an airline for dogs. 
* New York Times partners with Adelaide to track Ad engagement.  
* Scope3 introduces GMP+ a new product to prevent problematic ad placements.  
* Robinhood launches website for its new media platform, Sherwood News.   
* ASA shares their AI detection tool processed three million ads in 2023.   
* Accenture acquires integrated marketing group, Unlimited.   
* Hootsuite to acquire Talkwalker, AI social listening platform. 
* Clipboard expands its fediverse integration to over 400 creators. 
* ASA banned Nationwide Bank’s new TV Adverts for misleading public on bank closures. 
* Marks & Spencer named UK’s most prominent online grocery brand. 
* Specsavers launches a new Roblox Obstacle Course. 
* Blackboard launches new performance CTV ad platform. 
* Former Kraft Heinz CMO joins Bacardí. 
* SeenThis relaunches their Emissions tracking dashboard.  
* Apple TV’s market share grew in the US during Q1 2024.  


**Source: The Social Juice Newsletter.**
**Post 3:**

> What happened in marketing and social media last week? (Snap vs Pinterest)  
#Top 6 Updates of the Week: 

* Instagram users in U.S. can now create AI versions of themselves. 
* Snapchat reaches 850M users, 11 million premium subscribers in Q2 with ad revenue increase. 
* Google introduces new reporting and genAI tools to Performance Max. 
* Meta’s ad revenue in Q2 increased to $39 billion, thanks to AI.  
        * Family DAP (daily active people) was 3.27 billion on average for June 2024, an increase of 7% year-over-year.
* Pinterest added only 4 million new monthly active users in Q2, a huge drop in platform user growth. 
* Microsoft shares Q2 earnings, reports an 10% revenue increase for LinkedIn and strong user engagement growth. 

#Trending 

* Google Ads hit by major reporting glitch, exposing competitor data.  
* U.S. Senate passes landmark bills to protect kids online. 
* Pinterest and LinkedIn team up with Publishers to sell Ads. 
* Amazon Ads services revene grew 20% YoY in Q2, reaching $12.77 billion. 
* Google pulls back AI Gemini ad after backlash.  
        * Apple also pulled back their ‘OOO’ ad for being called out as ‘outdated and stereotypical’.
* Disney and Spotify partner with Yahoo for audio ads. 
* Outbrain confirms Teads purchase in a $1 billion deal.  
* LinkedIn introduces verification for company pages. 
* Snapchat publishes new study with National Research Group, sharing shopping insights, 82% snapchatters consider shopping to be a primary hobby, versus 59%. 
* British Library, Tate and 13 other museums in London and Oxford selects ‘Anything is Possible’ to handle their planning and buying.  
* Google introduces 3 new AI features to Chrome for product discovery.  
* US Senators propose new bill to combat AI deepfakes.  
* e.l.f. Cosmetics is celebrating Olympics with Hot Girl Walk championships. 
* Twitch mobile app gets a redesign, the new update puts their TikTok-style feed in the focus.  
* Snapchat adds new AR experiences for 2024 Olympics. 
* Vinted enlists Vicky Pattison to host new Channel 4 entertainment series. 
* Substack App introduces new mobile editor to write and publish posts. 
* John Lewis boosts retail media business with Epsilon partnership. 
* US Ad business lost 500 jobs in July as the labor market weakened. 
* Dash Drink’s brave OOH takes a stab at Coca Cola. 
* Reddit CEO says Microsoft and others need to pay to search the site. 
* Mischief US and Carter’s show messier side of childhood in back-to-school campaign. 

#TikTok 🎶 

* TikTok launches new app centre for businesses, providing easy access to marketing tools. 
* TikTok presents their first-ever Visionary Voices Africa list.  
* TikTok and CreatorIQ publish new report on Top 100 brands winning big with Creators.  
* Justice Department defends TikTok ban law, sharing TT collected U.S. users’s views on gun control, abortion and religion. 
* TikTok launches In-app hub for Olympics, it includes a medal table. 
* TikTok is spending $20 million a month at OpenAI through Microsoft. 
* ISpot.tv to provide cross-platform ad measurement with TikTok. 


 #Instagram & Threads 🗂️ 

* Threads App reaches 200M monthly active users. 
* Instagram is getting rid of ‘Posts count’ from Profile Page. 
* Threads releases TikTok-like blue search label to be seen above posts. 
* Instagram testing a new feed with reels that include notes from your friends. 
* Instagram Carousels got text overlays feature recently, now the feature includes new fonts. 
* Instagram now allows users to link their Spotify accounts for easy music discovery. 
* Instagram reels now have a prompts feature, next to polls. No idea what it does, prompts are show up just like captions. 
* IG Broadcast channel gets ‘imagine’ sticker feature. 

#Meta 😅

* Meta under investigation for running hundreds of ads for Cocaine, Opioids and other drugs.  
* Meta quietly scraps Celebrity AI chatbots as they fell flat with users.  
* Celebs are out, Business AI bots are In. Meta is developing AI bots to help businesses with transactional engagement. 
* Meta to pay $1.4 billion to settle Texas facial recognition data lawsuit.  
* Facebook’s U.S. Gen Z users are projected to increase from 49.0% (33.9M) in 2024 to 56.9% (40.5M) by 2028.  
* WhatsApp rolling out a feature to share voice messages with Meta AI. 


#X (Twitter) 🕹️

* Elon Musk accused of not labelling Deepfake video of Kamala Harris. 
* X aka Twitter disappeared from Mac App Store and then relaunched on macOS. 
* X expands Safety X support centre with new facility in Texas.  
* xAI considers buying Character AI. 
* X could soon launch ‘parody’ label for parody accounts. 
* Paris Olympic Medalists are getting affiliate badges. 
* Edit functionality for DMs is coming soon. 
* Elon Musk’s X suspended ‘White Dudes of Harris’ account and reinstated after backlash from Media. 


#Youtube 🕹️

* YouTube sports viewing soars as streaming moves to big-screen TVs. 
* YouTube announces changes to YPP suspension appeals and simplifies the channel page.  

#Google 🔦 
 
* Google Ads introduces new consent settings tab in Data manager. 
* Google Ads to let you set up message assets to click-to WhatsApp for some advertisers. 
* Google is showing X carousels less often for news topics. 
* Google Ads introduces new surcharges for specific jurisdictions. 
* Google updates image search features and shares on insights on how they handle deepfakes. 
* Google Discover promoting AI overviews for Olympic Games in Paris. 
* Google streamlines ad creation with Merchant Center-Ads integration. 
* Cloud expands its database portfolio with new AI capabilities. 
* New Google Business Profile AI tool creates a menu from an image. 
* Google introduces INP improvement for Publisher Tag Ads Library. 


#Agency News 

* ASOS hires Mother’s The Or as new ad agency. 
* Publicis Groupe hired as new media agency for Rocket Companies. 
* Krow Group hired as new lead creative agency for Benson for Beds. 
* Havas acquires Australian media agency and creative production company, Hotglue. 
* L’Oréal International distribution selects Wavemaker as EMEA Strategic Partner for Media Buying and Planning. 
* Spirit Airlines names Tombras Agency of Record. 
* Zaxbys appoints GS&P as new creative agency. 
* Ogilvy PR launches Influence Shield, a new offering to manage Influence Risk.  
* Hudson Rouge appointed as lead agency for Lobos 1707 super premium Tequila. 
* The AA appoints The Gate as CRM Agency. 
* McGarrah Jessee launches new social media agency, Look Squirrel. 
* Marketing Agency Myosin launches new platform that lets consumers give your media budget to Charity.  
* Stagwell revenue grew 6% YoY in Q2, thanks to new clients such as Chevy and Cadillac. 
* Corona Extra appoints GSD&M as the lead creative agency. 
* Boldspace appointed by Obrizum to rebrand AI-powered learning solutions.  
* VIVAIA appoints Edelman Australia as PR AOR. 
* Assembly appointed by Pipedrive, Tech Firm as global media partner. 
* Luton Airport Express selects Leith, Scottish agency as creative AOR. 
* ThePub adds content production mandate for TD in Canada. 

#AI 🤨

* WPP and Nvidia partner to generate 3D Ads using GenAI.  
* Google’s partnership with AI startup AThropic faces scrutiny from UK competition watchdog.  
* Software lab Forum3 launches Spok, an AI Marketing Expert. 
* Reddit acquires Memorable AI to enhance advertising performance. 
* Canva acquires Leonardo AI to boost its GenAI efforts. 
* Perplexity AI launches ad revenue sharing program for publishers. 
* Github launches GitHub models, a new generation of AI engineers. 
* OpenAI launches Advanced voice mode, currently rolling out to GPT Plus users. 

#Microsoft & LinkedIn


* LinkedIn is finally getting an easy to access ‘save’ button, available alongside engagement icons.  
* LinkedIn agrees to pay $6.6 million to settle advertisers’ lawsuit over alleged overcharges. 
* Microsoft makes Skype ad-free, good. 
* Nvidia and chip shares soar as Microsoft increases AI spending. 
* Microsoft's Azure networking took a worldwide tumble. 


#Marketing & AdTech 

* MiQ to acquire Pathlabs, expanding programmatic excellence to Independent Agencies. 
* impact (dot) com unveils fully customisable customer referral solution. 
* Discord on Xbox will soon let you watch your friends’ streams.  
* The Guardian issues apology after ‘watch porn’ ad displayed next to child abuse article.  
* Etsy to launch paid buyer membership program.  
* simpli.fi expands CTV Product suite, giving advertisers greater pricing flexibility and more. 
* NBCUniversal tops Olympics record as ad sales surpass $1.2B. 
* Experian rolls out retail media network solution.  
* For National Avocado Day, Chipotle and Wonderskin launched ‘Lipotle’ Lip stain. 
* Hyundai splits CMO role into creative performance positions. 
* Pacsun partnered with Pinterest for ‘Better in Baggy’ fall campaign.  
* Kiehl’s launches first-ever intimate care line: Kiehl’s personals.  
* Rokt partners with Vinted, powering new revenue opportunities.  
* IAS announces mobile in-app support for first attention product. 
* Twitch loses 10% of market share to YouTube and Kick. 

**If you liked this, follow me for more.**
**Post 4:**

> What happened in Social Media and Marketing last week? (LinkedIn changes) # Top 6 Updates of the Week:

* TikTok Ban is getting real, US House to vote this week on crackdown bill.
* Reddit announces new social and data toolkit for businesses, Reddit Pro.
* Google launches privacy-focused measurement tool for marketers.
* LinkedIn launches tools to assist in internal career progression.
* X launches long-form articles for all premium subscribers.
* FB, IG & Threads experienced global outages on Super Tuesday, Youtube & LinkedIn also experienced outages for a while.

# Trending

* Sam Altman says AI will handle 95% of marketing work done by agencies and creatives, do you believe the same?
* Pinterest launches new 2024 Wedding Trends Report.
* Telegram launches option to convert personal accounts into business, with new subscription model.
* Anthropic launches new Claude AI-3 models: Opus, Sonnet and Haiku.
* PETA’s nostalgic and jolly Cartoon Ad is trending but comments don’t believe it works to reach the Ad goal.
* McDonald’s Finland creates a Mockumentary trolling Website hamburgers are actually BigMac.
* Outsmart, US’s leading OOH Firm responds to a new report sharing roadside OOH ads are concentrated in areas with lower disposable income in England and Wales.
* ITV’s linear ad revenue declined by 15% in 2023 due to rise in digital revenue.
* The Ad Campaign for “The Gas Leaks Project”, parodies the shows like “Love is Blind” & “Love Island” to remind consumers of Natural Gas dangers.
* Pepsi Max officially launches its new brand identity in UK.
* ICO considers “consent & pay” cookie compliance proposal.

# TikTok 🎶

* TikTok expands monetization opportunities and rewards for Creators, pushing new resources too.
* TikTok Shop listings are surfacing on Google Shopping feed.
* TikTok launches new data portability API to comply with EU’s Digital Markets Act.
* TikTok goes fully into Oscars vibe with #FilmTok.
* TikTok’s new guide to reaching women on the platform with effective messaging.
* TikTok prompts the users in United States, to call the Congress and oppose the potential ban.
* TikTok testing new posting prompts that show up when you comment.

# Instagram & Threads 🗂️

* Instagram launches new and imporved Direct messaging features.
* Threads save a draft and Take a photo within app features are out now for all users.
* IG launches a hidden game in DMs for you to play when bored.
* Instagram testing auto-slide feature for Carousel content in the reels tab.
* IG prompting users to create more collaboration posts.
* Instagram events feature will be going away on March 19, 2024.

# Meta 😅

* Meta Engineering team highlights their changes to introduce messaging interoperability with third parties in EU.
* Meta now offers verified subscribers to make links more engaging with images.
* Meta provides new Tech transformations in 2024 & beyond.
* WhatsApp is testing addition of ‘Communities’ to Business App.
* New research shows that Temu paid Meta about $2 billion in 2023 for Ads.
* WhatsApp testing a new end-to-end encryption indicator for chats.
* WhatsApp users to soon get greater control over Avatars.

# X (Twitter) 🕹️

* X experimenting with new feed UI, removing all engagement metrics.
* X working on AI-generated content labels.
* X’s video and voice calling features are revealing user’s IP address to each other.
* X working on providing more advanced analytics.
* X’s plan to launch a Youtube TV clone for Samsung & Amazon Smart TVs.
* X now allows users to hide their creator subscriptions.

# Youtube 🕹️

* Youtube and Comscore expand their data partnership to add YT shorts measurement.
* Popular Science recruits Vsauce2 creators to relaunch its YT channel.
* Youtube interview MrBeast’s Thumbnail creator on their creator insider channel.

# Google 🔦

* Google March Core Update is here: 4 changes to link signal.
* Google updates search quality evaluator guidelines.
* Google Ads Country-Level Exclusions Limited To 120 Countries.
* Google March 2024 Spam update is also rolling out, alongside the core update.
* Google issues search ranking penalties through manual actions.
* Google complies with Digital Services Act, announcing new changes.
* Google is updating Local services Ads verification progress.
* GSC recipe rich result report adds two new properties.
* Google now redirects Business profile websites to Google Maps.
* Google Search People Cards going away on April 7.

# Agency News

* Spotify awards its global media account to a dedicated Publicis media unit, Onevibe.
* Monkey Shoulder hires Special London as global creative agency.
* Firehouse Subs launches creative agency review.
* BBC Studios selects
* as creative partner for debut US brand campaign.
* Amp unifies Five Specialized agencies into one integrated company.
* Uncommon Experience Studio wins Marshmallow Car Insurance (UK) account and launches first campaign.
* Spirit of Tasmania appoints CHEP Network as new creative agency.
* Havas Group achieves 4.4% growth for Financial Year 2023.
* Museum of Pop Culture appoints Iris as Activation-Engagement Agency.
* Shark Ninja appoints Horizon Next as Media AOR.
* TBWA\\Chiat\\Day NY appoints Emily Wilcox as new CEO.
* Independent Agencies Barkley and OKRP merge to form BarkleyOKRP.
* Gut is launching a design practice, to show their guts in a new service.
* Credit One Bank names Deutsch LA AOR.

# Brands & Ads 🏓

My Fav; Molson’s International Women’s Day  brand campaign is the best brand decision and marketing play gaining praise of everyone.

* Dove and Drew Barrymore confront issues of girls getting influenced by anti-aging online content.
* Peachaus unveils new brand campaign, building on Women’s confidence.
* SiriusXM launches first national campaign after brand refresh, it’s about dance & culture.
* In Japan, Apple brings 80’s Manga classic to reality using iPhone.
* Barnardco’s latest film visualises the idea of “kids living in their shell” for their charity program.
* For Guinness’s new campaign, Jason Momoa screams “I’m Irish” and he plans to do that this whole year.
* This Manscape Ad Campaign is for the boys and men too, but there is a difference.
* Old Navy counts on brand nostalgia in their first TV campaign of 2024.
* Adobe and Women’s FA Cup look to future generations.
* South Western Railways turns insults into an Anti-abuse Ad campaign.
* LEGO’s new Ad Campaign asks little girls to ditch perfect.
* GambleAware’s Smoke-filled Billboard tackles gambling stigma.
* Domino’s new Ad campaign is short and focused on comfort food and delivery trust.
* Heinz launches World’s first ever Ketchup Insurance policy.
* Sweaty Betty rebrands ditching their boutique branding for a bold modern look.
* StreetEasy Campaign turns the story of buying an House in NYC into an Epic Odyssey.

# AI 🤨

* OpenAI responds to Elon Musk with all receipts needed.
* Cloudflare announces firewall for AI.
* IAS expands “Made for Advertising” AI-driven measurement and optimisation solution.
* Salesforce Spring 2024 Release includes Einstein AI updates.
* DiGiorno’s launch new AI microsite that offers you discounts by scanning pizza delivery damages.
* Sam Altman rejoins the OpenAI board.

# Pinterest & Reddit

* Pinterest shares new guide to inclusive content for marketers.
* Reddit rolling out new AI tools to stop on-platform harassment.

# Microsoft & LinkedIn

* Microsoft Advertising Performance Max globally available now.
* Microsoft relaunches deep-search after the initial launch failure.
* Microsoft Ads launch new tools to make cross-border advertising easier.
* LinkedIn’s latest report shares most In-demand jobs on the platform.
* LinkedIn now allows you to filter “who viewed your profile” results by Interesting viewers filter.
* LinkedIn’s plan to start focusing again on news content and knowledge sharing.

# Marketing & AdTech

* Linktree partners with Sephora, Urban Outfitters and more to jump into social commerce.
* Sainsbury’s allows advertisers to create ‘branded shops’ online.
* New Congressional Bill would set minimum royalty payment at 1 cent per stream.
* Reckitt turns to livestreaming to sell condoms in China.
* STV’s Pitts named as new Global group CEO.
* Bloom appoints new executive leadership team.
* Disney sells out Oscar’s Ad Inventory, the early sales show new brands are here.
* Whirlpool helps singles clean up their dating profiles after finding, 91% of open singles prioritise someone that is able to do basic household chores.
* ASA changes the style of ban on Calvin Klein’s FKA Twigs Ad + bans a new BBH Ad Spot.
* Target to launch a paid membership program to compete with Amazon & Walmart.
* Roblox expands advertising options for brands with launch of new site.
* KPMG fined over M&C Saatchi audit failings.
* Corona appoints Pedro Pascal the lead brand ambassador.
* Accenture acquires The Lumery, An Australian MarTech Consultancy.

***I hope this helps you to plan your week ahead, you can find sources for all headlines in the social juice newsletter.***
**Post 5:**

> What do you think about a social media that revolves around informative content, investigative reporting, whistleblowers, cultivates communities around these topics, and incentivizes critical thinking? Is it possible? It's quite apparent (I think) that news shared online today is riddled with misinformation, propaganda, and opinion-pieces stated as objective truths. And because the news is being shared more and more on social media nowadays, its shared amidst unrelated brainrot content, which doesn't help. To 'combat' this, platforms have decided to moderate content by labeling things true or false and censoring the rest. This is an obviously impossible task and also nonsensical in that the truth is often too ambiguous to be labeled one way or the other. What we've gotten as a result is a world where nobody can reach consensus on anything and we're divided on the most simple issues.

I believe that social media could be a positive to our society if designed correctly. So I've been thinking more and more about this for a few months and started to take it seriously this summer. I'm hoping that by asking here I could get some general opinions as to whether this could work or not. I'm also going to try and be as short as possible so I don't make this a complete essay, but can hopefully elaborate in the comments if need be. Here's the rough idea thus far:

It'd center around short-form video content, similar to TikTok or IG Reels. This is because I do believe it's the most engaging for the layman and we're generally trending towards more video content and less text. The content would be focused on topics from local to global politics, business, sports, science, art, history, pop-culture, and so on. These topics I classify as "news," so when I refer to news I don't just mean things like CNN or Fox. There'd be no "brainrot" content per se, though people could post their opinions on things and such which I'm sure could, at times, fall under that classification. A machine-learning algorithm would have to be designed to present users with content of interest, but also avoid feeding similar stories and enabling confirmation bias.

There'd be a few main feeds:  
***"For You"*** which at this point is self-explanatory.

***"Local"*** which would contain a proximity bar for users to filter news nearby them. This would be especially beneficial I believe to local news producers and more local communities which are currently being erased by national and global corporations.

***"Whistleblowers"*** which is self-explanatory. I was thinking we (the platform) could have in-house journalists dedicated to receiving and vetting whistleblower testimonies and documents. Then, once it is received and vetted we create a short video design to summarize the findings, which would link to a special profile created for each whistleblower (which would obviously be anonymous), where the user would find a database of all the documents, etc.  
I was also thinking perhaps there could be an "unverified" section to the whistleblower feed where users  could upload anonymously, having their voices scrambled, no link to their profile, and such. This could allow for more "unserious" whistleblower content, or frankly just allow for people to rat out weird shit that they may have seen at work or elsewhere. It'd sorta be like the website "Fucked Company" that was created post dot-com bubble where employees posted anonymous comments about their employer. There's an obvious issue here of people being able to spread lies anonymously, which, yeah, would be a problem, but people can do that on the internet anyway.

Then, people would be able to create groups:

***"Newsrooms"*** would essentially be like subreddits. People would be able to create and moderate these rooms themselves and revolve them around certain topics or subjects.  
The main differences are that (1) it would revolve around people posting video content to a dedicated feed for that Newsroom, (2) people would be able to upload PDF documents to the group (depending on the moderator's decision) where they could then upvote/downvote and comment on the PDFs themselves, (3) there would be voice channels, like Discord, and (4) people could have specific roles within the Newsroom, also like Discord.  
I was thinking about having two separate applications (one for mobile, one for PC) where on the PC one there would be text channels, (also like Discord. Yes- I've used Discord a lot and got inspiration from it) which would likely be used by the more serious user of the platform, but for the mobile app it'd be just video.

***"Investigations"*** would be essentially the same as Newsrooms, except, obviously, for investigations. People could formulate and display their hypothesis, and build a community around that. For instance, a person could investigate Epstein in some way, have a hypothesis, and share their progress and receive feedback with the community they've built. The design of the Investigation room would be unique to reflect the nature of hypothesis testing, though I'm not sure exactly what that'd be as of yet.  
Also, there could be a crowdfunding feature where the person could raise money to support their investigation. Nowadays investigative reporters have to rely on corrupt establishments to fund them and naturally are going to restrict that investigation in many ways. Perhaps there could even be a feature where we (the platform) fund investigations that have reached a certain popularity.

Now, that'd be the bulk of the design. Here would be some features for within the app:

**(1)** People can cite PDFs and links to their video uploads. (These citations would be stored somewhere on their profile as well)

**(2)** There would be a "Learn More" feature attached to every video, where a user could click and receive an AI prompt that would do something of the following: *(1) do sentiment analysis of the video and source counterpoints (2) direct a user to more information regarding the topic discussed (3) display 'provable facts' regarding the topic discussed, etc.*  
Maybe it'd do a few other things, but the idea for this feature is that it'd allow for people to "jump down the rabbit hole" on topics and discover the truth. Currently, we're presented with a video on a topic but aren't necessarily directed to more information. If we are, the creator of the video would direct you to more information they've also created. Thus, the information is being shared in a vacuum and people aren't receiving other opinions or sets of facts.

For this feature, users would gain +1 point (sorta like karma) for each time they clicked the feature, and it'd be displayed on their profile. In addition, if a person posted a video and, say, 10,000 people clicked learn more on it, they'd receive +10,000 points. For this set of points, perhaps there'd be a financial payout of say $1/1,000 clicks, or something of that sort. This way creators would have a financial incentive for their viewers to learn more about the topic they're discussing.

**(3)** Users could run polls in the app. The creator of the poll could put some requirements on who they let answer. So, for example, there'd be a place for typical users to upload their real ID to verify their account, and a person creating the poll could limit answers from people with verified ID, living in the state of New York.

**(4)** Voice channels. I briefly mentioned it but I think it's a feature worth describing in a bit more detail. People could join these channels to discuss issues, even host 'live' podcasts, where users can listen in (twitter does a similar thing now, though I don't like the UI of it). Every profile would also have a dedicated voice channel that would activate when that user joined it, and they could make it open to the public for people to join, restrict access to certain users, or make it completely private but allow users to "listen in". A politician could in theory join their dedicated voice channel and host a "fireside chat" where people can listen into. Pretty much radio.

Anyway, this is the rough idea. I want to create this thing with the opinions of the people who may use it in mind, hence me asking about it here. I think the world of social media is fucked up and I'd like for it to be better. Also, for anyone who may ask, yes, there would be ads. I think having no ads is a nice idea, however, I'd like for the platform to be free to use and also have financial payouts for the people who post on it, which would only be funded by ads. FYI, I'm currently an undergraduate at the University of Pennsylvania, so I've got lots of ideas but little experience, another reason I'm asking here. Also, I've likely omitted something, so if something doesn't make sense, please ask.

edit: Added features (3) and (4) which I initially forgot. Also forgot to mention there would be a machine learning algorithm.
## Topic 2: Influencers & Platforms
**Post 1:**

> South Australia Minor Candidates Debate Summary On 19 March 6 News hosted a debate between South Australian candidates with a realistic shot at the final Senate seat. This included One Nation, Jacqui Lambie Network (JLN), Family First and a candidate formerly of Legalise Cannabis. This is a summary of what happened for those who don't want to watch an hour and a half, posted on Sunday because that's the day for personal posts such as these. It's a week late because writing these is several thousand words, several hours of watching and editing and a fair bit of research, and you are not getting more than one a week.

As previously:

[full livestream](https://youtu.be/R8fLagREPB8)

[live blog from the guy who did the live blogs](https://bsky.app/profile/anthonyzougras.bsky.social/post/3lkpr57tanc2p)

And if there are any more debates as have been teased, I will endeavour to do similar!


-	Has a special guest; my fiancée!  She is in her mid 20s and has little interest in politics but watched it with me because she knew I was interested.  I will also be including her opinions of the candidates because as a person nearly completely divorced from politics she’s probably a better example of how the average person sees this than I am.  Overall, she thought the debate lacked passion and was not too enthused by any of the candidates. 

-	Easily the most sophisticated of the debates; actual policy kept being brought up as opposed to ideological concepts.  If this is due to the general demeanour of the candidates (all four are veteran political campaigners), South Australia being a less polarised environment or if this is the old free settler stereotype manifesting, who knows.

-	Marked political shift from Queensland especially towards the centre.  Brohier attempted to argue that renewable energy drives up power prices… and got torn apart by both Patrick and Walter, with both being polite but quite clear they think he’s wrong and uneducated on the topic.  Meanwhile, Queensland had a solid 75% majority who didn’t believe climate change was real.

-	The only debate to really get into niche issues only relevant to the state.  I’m sure absolutely nobody outside of South Australia wanted to hear about the Murray-Darling Basin Plan tonight, but the candidates decided you were going to be learning all about water buybacks anyway.  Even as a resident of the state who votes primarily on the environment, I had to look up some of what the candidates meant.

-	Both Brohier and Game were absent from the debate for substantial chunks of time due to internet issues, which I will take into account (to an extent, as I think it’s also reasonable to expect candidates to have a stable connection in 2025).  I’m also somewhat convinced Game fell asleep through the start of the debate and that’s why she was late.

-	Dianah Walter used to be the Legalise Cannabis candidate but left because of what a situation that according to her, forced her to choose between the party and loyalty to the state.  Nobody asked her what this issue was which is, to me at least, frustrating.

-	This debate was generally poorly viewed, with less than 50 people in the stream and only 1/3 to ¼ the views of the other two streams.  The highlight has less than five hundred views, compared to 3,200 for Victoria and 29,000 for Queensland.  So despite being overall higher quality, it mattered even less.

-	Full disclaimer: as an SA voter, my preference order would be Patrick, Walter, Game and Brohier.

**Rex Patrick, Jacqui Lambie Network**

-	Senator for SA between 2017 and 2022, both for the Nick Xenophon Team and as an independent in all but name.  Was previously Xenophon’s chief of staff, and took his seat when he resigned to run for a seat in SA parliament.  Received 2% of the vote in 2022.  That is to say, he has the most experience of anyone here in being a politician and will be judged accordingly.  

-	Has worked in transparency since the election but prior to his political career was a submariner, then a military contractor.

-	Rex has a problem that isn’t really his fault, that being that he just *looks* like a conservative.  He has the physical appearance, clothing style, mannerisms, accent and behaviour of a 60-something from Whyalla (which he is), and it’s going to turn people off him on appearance alone because his politics aren’t what you’d expect from someone who looks like that.  The Lambie bomber jacket, while a good way of communicating the party you’re a part of, isn’t helping.  It’s hard for me to explain to a non-South Australian, but he just looks like the kind of person that votes Liberal but doesn’t follow politics at all.  Hearing a guy that looks like him say “LGBTIQ” is just… strange.

-	His background is just his office, which actually kind of works because he’s got a massive bookshelf which makes him look educated.  There’s a Father’s Day card and pictures of what are presumably his daughters, which is cute.  However, there’s also three (unopened) bottles of wine hanging out up there which might be there for decorative purposes (one of them is a Penfolds I think) but is a bit suspicious.  Webcam quality is okay but not exceptional.

-	My fiancée thought it was hilarious that he had a bunch of what appeared to be the same big red book and wondered why he had so many.  We both also agreed he looks like a family member’s ex-partner who is also from the Eyre Peninsula and around the same age.

-	World record stammerer.  His biggest weakness in the debate to me was the amount of ums and ahs that came in between what were usually quite punchy points.  It was clear that Patrick had a depth of knowledge over many of the areas, but his manner of speaking made this harder to explain.  I don’t know if it’s a lack of preparation or just how he talks but it wasn’t good.

-	Astonishingly persistent.  When an opposing candidate gave an answer he considered incomplete or incorrect, he would go after them and pin them down until they answered or the moderator intervened.  

-	Immense depth of knowledge on South Australia, the military and gas especially, all areas he focussed on in politics.  He showed a greater level of knowledge in his fields than any other politician in these debates.  When anyone challenged him on these topics, he was unassailable.

-	The three factors combined means he just talked a lot, more than any other candidate.  This let him dominate the debate to an extent, but also means more writing for me.

-	Much like Lambie herself, he’s weakest on topics where her base is split or issues where the “party” (JLN at a federal level is Lambie doing her own thing, in effect) doesn’t have a position.  Aside from a general libertarian vibe, they just don’t have concrete positions on issues like if minors should be allowed to transition.  He’s effectively a regionalist voices for SA and unlike Walter, he was unwilling to outright call these kinds of policy areas a distraction.

-	Introductory statement is for “common sense” and “not about left or right, right or wrong”.  It isn’t a very good statement, in my opinion.

-	Believes that having an independent Senator from SA is important.  This gets repeated a lot by both him and Walter.

-	Claims electricity is too expensive and is set by gas prices.  Claims that our gas is being exported overseas, which drives up prices.  Patrick wants Australian producers to be prioritised and hints at wanting a gas reservation policy simply to WA.

-	Wants to ban foreigners from “investing in housing” (am unsure if he means banning them from owning all houses, or houses they don’t live in)

-	Supports divestiture power for supermarkets if they are found to engage in a monopoly.  Part of he and Brohier’s arguments talk about this and it gets into slightly nitty gritty detail, but the simplified argument is that these powers will act as a deterrent to prevent monopoly behaviour.

-	Cuts through Brohier’s argument to state that the gas price is the primary contributor to gas prices, and that “no-one would question that”, to which Brohier responds by questioning that (or by straight up saying he thinks Patrick is wrong, rather).  He also argues that renewables have not causes the electricity price increases in South Australia.

-	Part of the stream is cut off, but Patrick believes that climate change is real, renewables are the cheapest energy option for SA and when Brohier denies climate change is a problem says “you have a choice of trusting CSIRO, or Family First”.  It’s a great line and the way he’s smirking he knows it, but it bothers me that Patrick pronounces CSIRO as one word rather than an acronym, because I’ve never seen anyone else do that.

-	Wants to see immigration “slowed” from current levels, and believes some current immigration is to “inflate GDP”.  Wants immigration matched to skills (it already is…) and specifically more immigrants to regional SA.  Wants an immigration plan that is integrated with our plans for other services.

-	Believes immigrants benefit the country, specifically citing food and seeing diverse shops.

-	Puglisi states he is “critical of AUKUS”

-	Believes that the Trump administration is not normal and are acting in ways against Australia’s values.

-	Doesn’t want increased defence spending until we show greater fiscal prudence, citing the Hunter-class and Attack-class as specific failures of spending, or we’ll be unable to spend that new money properly either.

-	Absolutely massive argument with Brohier that 6News chose as the highlight clip for the debate.  Calls him an idiot and dangerous for his suggestions, and then goes straight into the second debate when Brohier claims that Australia was unable to respond to the Chinese fleet near our coast recently.  He immediately goes in depth about our arrangements with New Zealand, the ships involved, freedom of navigation and how you might monitor a fleet in this situation.

-	Believes that the Chinese circumnavigation was not unlawful and that similar circumnavigations are done all the time in the South China Sea.

-	Generally accepting of people’s gender orientation, but in a bit “everyone should respect each other, and I won’t offer my own opinion” kind of way.

-	Abortion is a state issue, but he has no issues with SA’s legislation.  

-	Supports same-sex marriage as he thinks it harms nobody and allows Australians to be happy.

-	Is asked by Puglisi if he can commit to remaining within JLN if elected.  Patrick says he can, after a fairly weak explanation (that NXT became Centre Alliance and he only left the party after Xenophon left politics.  This is true, but he also had no chance of being re-elected as he would have been the second senate candidate behind Stirling Griff for Centre Alliance).  Says that they have different views in some ways but get along well due to their friendship and combined background.  They apparently have a “nice and flexible policy” together.  

-	Says that he only ran because Lambie kept nagging him to do it (from reading up, apparently she asked him for about a year before he finally gave in, including during the Tasmanian election period).

-	Patrick asks his one question to Jennifer Game, asking if she would cross the floor if One Nation took a position on the Murray-Darling against SA’s best interests.  She refuses to answer as it is a hypothetical question.  He says that it’s a pretty simple question and that One Nation took positions on the river that were against (in his view) SA during his time in parliament (which I had to google; what on earth is a sustainable development limit).  She just refuses to answer and eventually he gives up.

-	She then asks him about if he agrees with something Jacqui Lambie said during the COVID era, that being that everyone should be vaccinated.  He says that he’s not up to date on the science and effectively cops out of the question just like she did, so we don’t find out his opinion on vaccination.  He says “he will go with the science”.  

-	She then comes back for her hat by saying that she won’t ask him about crossing the floor as it’s also a hypothetical, to which he responds that he and Lambie do have an agreement in place for this!  He claims that if an issue affects SA and Tasmania differently they will vote differently, and if it does not affect one state (such as the Murray-Darling not running through Tasmania) one will vote for the other, which he claims is effectively a bonus vote for both sides on South Australian/Tasmanian issues.  This is actually quite an interesting idea and I wish it was brought up earlier.

-	Cites the Make Australia Make Again policy, which involves exporting more manufacturing goods instead of raw materials.  He specifically cites iron ore (which is likely a reference to the Whyalla Steelworks) and lithium batteries.

-	Will “keep the bastards honest” (yes, he said the quote).  South Australia arguably invented the Democrats and JLN is a centre party using the colour yellow so it kind of works.

-	Rex unironically calls himself the transparency warrior (it’s in all caps on his website too) in his closing statement and I know he’s a national expert on FOI, but it’s such a stupid name and makes him sound so, so silly. 

**Jennifer Game, Pauline Hanson’s One Nation**

-	An experienced political candidate, who was their candidate in 2022 and whose daughter Sarah is the only One Nation MLA in SA.  She has also (apparently) worked for Hanson since 2015.  She received 4% of the vote in that election, enough that in a double dissolution she would likely have been elected.  

-	She’s the most difficult to score in this list by far, as while she was very capable, she arrived 20 minutes late to the debate and it took her another 3 minutes to turn the light on in her room.  My fiancée and I both thought that she might have slept though the start of the debate as the light was off and she was in very casual clothes, though there’s no proof on that and it could have simply been network issues.  Once she’s online and switches the light on, the webcam is of acceptable quality though the background is just her room.

-	She’s quite eloquent and makes strong arguments.  She has the traditional South Australian accent which helps, but I’ve seen the One Nation party line three times now and Game absolutely blows the other two candidates I’ve seen away.  She just has better delivery and depth of knowledge, actually citing statistics to back up her claims, and is far better at seeing and dodging potential issues. 
 
-	I appreciate her listing policies but doing it in the closing statement at the very end of the debate is not how you do it.  It means we didn’t see a debate on any of the items listed, which might have been interesting!

-	Believes we should reduce immigration.  Claims that we would need to double our build rates to fit all the migrants coming into the state at present rates.  Wants around 130,000-150,000 migrants annually, as per the One Nation line (they want migration equal to the amount of people departing).

-	Wants migrants who assimilate well and contribute to the country.

-	Says that we don’t know if we can rely on the US as an ally.

-	Claims that Trump has a “very reasonable position” on Australia’s defence, and that he wants us to escalate our spending to avoid the US having to do the work.

-	Party position is that gender fluidity is real, but that there are only two sexes.  Says that they accept people “regardless of how they want to be.”  Immediately pivots to gender “indoctrination” in schools.  Claims that there are attempts to “interfere and sexualise our children” within schools.

-	Is asked by Puglisi if she would stay in the party if elected, considering One Nation’s history of defecting members at a state level.  She says that if you stand for a party in the Senate you should stay, and if you find that untenable you should resign rather than leaving the party.  Says she has worked with Hanson for a long time and that it won’t happen.

-	Claims that her daughter has managed to achieve several pieces of legislation despite being a single member through bringing it to state attention, and that therefore single members can be useful.

-	Does not support further water buybacks.

-	Patrick asks his question to her and while it’s on the Murray-Darling, it’s really about what she would do when what’s best for SA and what the party’s policy is diverge.  She effectively says they’d debate and decide what is best collaboratively.

-	Her question to Patrick is about if he agreed with Lambie that “everybody should be vaccinated”.  While the question is a little odd (SA had one COVID lockdown in 2021, it just isn’t the same issue here as it is in Victoria) it is an excellent question in that Patrick doesn’t really have an answer.  It is implied through her statements (and previous actions) that she does not support people being required to be vaccinated.  As mentioned above she then fumbles what was otherwise an excellent attack that really left him wrong-footed.

-	Closing statement mentions that we should focus more on young people who aren’t in employment or training.

-	Rapid-fire closing statement policies:

-	Income-splitting for families, so they can reduce their taxes

-	Removing GST from insurance

-	Halving fuel excise permanently

-	Increasing tax-free threshold for self-funded retirees to $35,000 (had to look up the details on that one)

-	Allow retirees to work more before having pensions affected.

-	Small government good, government waste bad, freedom good.

-	Believes government debt is inflationary and “another form of taxation”.

**Dianah Walter, “SA Matters”**

-	Did a little bit of research and she is a former staffer for Geoff Brock who ran for the seat of Narnugga (Fraser Ellis’ seat) in 2022, as well as a Senate candidate in 2013.  She is also the founder of Regions Matter, “a micro enterprise established to help governments, the private sector and not-for-profit organisations engage with their respective stakeholders and communities, and vice-versa.”

-	I believe she has worked in the medical field, but finding out what she has done is quite difficult. 

-	As she is an experienced campaigner, I will say that while her background is polished and so is her speech, her webcam is very suboptimal.  I also did not like her lack of specific policies, as opposed to focussing on general areas of development, especially as she has run before.  Part of this is likely due to her running originally for Legalise Cannabis, but I still do not like it.

-	My fiancée thought she had weird movements, but otherwise did not like or dislike her.

-	Very patriotic for South Australia.  Talks at length about her passion for bread and butter issues for the state.

-	Believes that climate change is real and that “renewables are the way to go”.  Cites her experience in regional SA, as well as the private sector doing the same regardless of government incentives.  Supports South Australia’s highly renewable grid (presently about 70% of power in SA is renewable).

-	Does not believe immigration is causing the CoL crisis or housing.  Claims that immigrants are being brought in to do jobs that native Australians don’t want to do, and that they’re a vital part of the economy.

-	Says that while the US alliance is somewhat important and South Australians broadly disapprove of Trump, senators should be focussing more on local issues of greater importance to voters.

-	Supports having closer relations with Indonesia specifically and the Pacific nations generally.

-	Wants a bigger focus on veterans and soldier welfare while in the ADF.

-	Wants people to be kind and respectful to one another regardless of religion.

-	Supports voluntary euthanasia.

-	Thinks that most South Australians are not concerned with if children are or are not trans/gender fluid/non-binary etc, and they have more important concerns.

-	Thinks that more attention should be given to women’s health, citing specific examples of areas that need care from her experience.

-	Supports increased money in all care, including preventative care and mental health.

-	She still supports legalising cannabis, despite no longer being a member of the party.  Also supports the use of cannabis for children, but only in a medical context.

-	Wants to develop an SA hemp industry.

-	Did not have a final question for any members of the panel, which to me is poor preparation.

-	Supports raising the rate of Centrelink payments (I believe she’s referring to Jobseeker specifically).

**Christopher Brohier, Family First**

-	It’s impossible to discuss Brohier’s performance without mentioning his sheer insistence on trying to pick fights on topics that Patrick is an expert on.  Basic research or simply paying attention to SA politics between 2017 and 2022 as I’d expect a former lobbyist to have done, would have shown the folly immediately.  He doesn’t win a single one; it’s an insane strategy to argue with someone who is known for being persistent on topics he has far more experience on, and yet Brohier keeps doing it again and again.  In terms of oratory every person who’s been in a parliament (bar Roberts) has been a more eloquent and fluid speaker, but Patrick is deeply knowledgeable and willing to go in for the kill in a way that nobody except maybe Rennick has been, and Brohier kept offering himself up on a silver platter.

-	Is a barrister and the former director of the SA branch of the Australian Christian Lobby.  He brought up that he’d been a barrister for 40 years several times.

-	Managed to achieve the feat of having my fiancée actively dislike him by the end, a unique distinction.  While she generally considered the candidates not great, he was definitely the one she liked the least. 

-	His signal wasn’t great which means at times he cut out.

-	Brohier kept getting way, way too close to the camera, to the point that sometimes not even all of his head was visible.  His webcam quality wasn’t bad (with a Family First background, albeit covered up by the news ticker) and he spoke eloquently, but towards the end he kept all but eating the camera.

-	Three key policies in the opening statement.  They are:

-	Reducing CoL by “attacking power prices”. 

-	Opposing minors transitioning.  They also do not want “the promotion of gender fluid ideology in schools to young kids”.

-	Opposing faith-based schools being considered under discrimination laws.

-	Claims that the “real cost of energy has gone up 65%” between 2007 and 2021, and that renewables have caused this.  Wants “dense” power, including potentially nuclear.   Argues that it “can’t” be true that renewables are cheaper due to the high transmission costs and firming.  He specifically argues for gas and coal-fired power stations.

-	Says “the climate catastrophism just isn’t happening”.  At this point Patrick goes to reply and Puglisi jumps in, as Walter has been waiting patiently while the two go at it for several minutes.  She then not-so-subtly implies he doesn’t know anything about regional SA or power.  After a little more back and forth they move away from the topic.

-	Claims that Woolworths and Coles do not have a monopoly.  He provokes a fight with Patrick over the JLN’s policy to have a law to break them up which as a lawyer he should win.  However, Patrick does a good job in explaining how it has a deterrence value and then can cite him personally attempting to introduce a bill for this, as well as citing support.

-	Wants immigration closer to pre-COVID levels of around 200,000 per annum.  Believes this will help housing.

-	Ambivalent about the US’ strength as an ally and wants to build up an independent capability.

-	Claims that DEI is why the ADF doesn’t meet its recruiting targets, and that they are not recruiting on merit.  This is not an exaggeration.  He argues that the US Army has gotten more recruits since abandoning “DEI” and that even if they do recruit on merit, the optics are that they do not.  He then ends up in an argument with Patrick again, who is the only person on the panel with a military background, on military recruitment.

-	Claims that China sailing a convoy near our waters and (he claims) our inability to track them was a sign of our weakness.  Patrick immediately disputes his claims using his decades of experience in the Navy and military contracting, and it devolves into another debate until Puglisi moves the topic on.

-	Is asked about how he and the party represent non-Christians.  He says that he will represent people based on the dignity of the individual, and that God does not accept forcing your beliefs on others.

-	Disapproves of South Australia’s abortion laws (abortion up to 28 weeks in cases where it affects the mother’s mental or physical health), and claims they had a poll that said so in 2021.  Claims this policy is “brutal” and that babies are born alive.

-	Wants support for mothers so that “abortion isn’t the only option” (a paraphrase)

-	Puglisi (the moderator) cites Family First’s manifesto, which argues for the reversing of gay marriage as a policy.   Brohier says that he didn’t know Lyle Shelton said that and that instead they want to support straight marriage instead.  He believes we have to "live with the consequences" and that it won't happen by 2025.

-	Claims that straight marriage is the best kind of marriage and that gay marriage led to gender fluidity (being accepted I assume he means here).

-	Is the only person in any debate so far (12 candidates) who does not support legalising cannabis in any form.  Claims that it can be harmful for children and that there is no research on dependency.
**Post 2:**

> Victoria Minor Candidates Debate Summary On 12 March 6 News hosted a debate between Victorian candidates with a realistic shot at the final Senate seat. This included One Nation, Legalise Cannabis, the Victorian Socialists and the Libertarian candidates.  This is a summary of what happened for those who don't want to watch an hour and a half, posted on Sunday because that's the day for personal posts such as these.  This debate was overall slightly more mature and as a result there's a lot more policy for me to cover.  South Australia will follow next week if there is as much interest in this one as there was for Queensland, as well as any future debates 6News sets up.


[The debate itself, if you want to watch it all](https://www.youtube.com/watch?v=5xpd8JKYGSs)

[Link to the only guy who was reporting on these that I could find](https://bsky.app/profile/anthonyzougras.bsky.social/post/3lk6alxaamx2j), just like last time.  Please can someone with a Bluesky watch the next ones.

**General observations**:
-	Because one of the debaters was an e-celebrity, the audience was always going to be majority pingers fans regardless of who was the best at debating or had the best points, because they’re there to see their guy.  Very clearly a lot of pingers fans but also a surprising amount of Libertarians in the audience.  I don’t think anyone will be surprised that pingers won in terms of the after-debate poll.

-	Fiona Patten won in terms of actual debating ability, but it doesn’t matter.  These types of debates, considering the sheer diversity of views on display, are extremely difficult to “win” in the public eye by being centrist and reasonable when there’s both a left and a right element present, and the audience wants extremes.  It’s also completely expected for her to be capable of pulling that off against two people with no political experience and a social media guy; she was an MLA for 8 years.  

-	More broadly, I really don’t know if there was much of a point at all between people so far apart debating, as neither side is going to convince the other and the gap is so big that anyone in the “middle” has a good dozen parties to pick from instead.  It is on some level good entertainment.  Nobody got a real shot away either, as while the debate itself has more views than Queensland, the Queensland highlights reel has an order of magnitude more than the Victorian highlights.

-	Nobody was genuinely awful in any way, or even really bad.  I hold Patten and pingers to higher standards because of their experience, but while nobody reached the stratospheric heights of Rennick in the last debate (whose debate highlight is the 9th most viewed video on their channel and is probably the only person to have gained many votes from this exercise) the SA and QLD debates definitely had far worse efforts.

-	The debate on the forbidden topic involved one side that sincerely believed the other was supporting terrorists and terrorism, and another that sincerely believed their opposites supported, or at least were neutral on, genocide.  Considering this, it was surprisingly polite.  I will generally avoid discussing the part of the debate covering that topic because I too think it’s devoid of value, but considering just how far the two sides are and the magnitude of the issues they have with their opponent, it’s remarkable that only a few swear words got thrown and nobody got truly angry.

-	Universal agreement that nobody thought their position was extreme, at least with proper historical and political context. 

-	Credit to Puglisi and his comedic chops for this question; “Mr Dittloff (Libertarian), do you think landlords are productive members of society?”  One guess at the response.


**Jordan Dittloff, Libertarian**

-	Impeccable dress, to the point that it almost looks like he’s overdressed.  Professional quality webcam, full suit and the only candidate so far with a Teams background, this being the Libertarian Party logo (update: Dianah Walter also had her logo, but her webcam was worse).  While he looks like someone who is obsessed with day trading, it is also an extremely professional look which as someone with no MLA or media experience is probably the way to go.

-	Clearly had done at least some reading on his opponent, and identified areas where both he and pingers agreed government spending should be reduced as well as coming pre-prepared with a question for Patten that she was unable to respond well to.

-	It’s Victoria; did you REALLY expect we’d get away without talking about COVID?  Pingers (why do you only talk about cannabis and not people’s wider lives) and Pickering (why do you preference Labor and the Greens, as in his opinion it’s in their interest to not legalise cannabis to keep that preference flow going) both take shots at Patten, but Dittloff is the one that has impact.  He asks her about voting for pandemic emergency measures in 2022, which helped secure the passage of the bill.  Her response is pretty wishy-washy, arguing that the formalised emergency powers helped provide a check on the government 

-	He does get away a little bit when nobody takes up his claim that COVID “wasn’t a deadly disease”, considering the deaths it caused even with lockdowns.

-	He straight up saluted Puglisi when his name was called out, which I thought was funny.

-	Begins with their ideological beliefs in small government, specifically saying that he/the party consider the Liberals as the least bad, but they no longer fit Menzies’ ideals.  Very brief opening statement, no policies mentioned.

-	“We don’t have a cost of living crisis, we have a cost of government crisis”, which as both him and Willmott repeated it verbatim, really does make me think that they had that line set up to say in advance.

-	Believes that the Cost of Living crisis is caused by government overreach, including areas that have “never previously been the purview of government”.  Claims that 83% of new jobs have been “government-funded jobs” (unsure how broad this is; could mean anything from public servants up to any job in an industry with government funding)

-	Wants the fuel and alcohol and tobacco excise gone, and alludes to increasing rates of black market cigarettes as an example of their failure.  Claims they are a “tax on the poor”.

-	Tax brackets gone, flat 20% income tax.

-	Implied that he does not support axing the NDIS entirely, selling the NBN or defunding abortions.

-	He wants to defund the ABC and abolish the eSafety Commission, cut all aid to Ukraine and all support for either side over the war in Gaza.  

-	Claims that the Department of Education would have benefits, as it has no teachers and “contributes nothing of value”.

-	As alluded to, believes that landlords are investing capital into an investment with risk, and that the market has incentives for landlords to find and keep tenants, instead of empty houses.  Believes that Victorian tenancy laws are too strict, and that is causing people to leave homes unoccupied to avoid having a tenant.  Specific examples he gives are allowing pets and allowing them to paint the house whatever colour they choose, and that these give them rights that are too close to owners.  

-	Agrees with pingers that landlords can be bad people but unfortunately the audio is horrible so I can’t tell you anything else. 

-	Supports withdrawing from the International Covenant on Economic, Social and Cultural Rights, which in the context it was brought up involves housing standards. 

-	Believes that “migration… should generate a benefit for the existing society”.

-	Wants an immigration “auction” where a certain number of slots are given per month based on a percentage of the population (1 percent per annum), and they can be bought by employers or individuals for any reason.  Is implied he wants to get rid of at least immigration from the Department of Home Affairs.  

-	Claims that subsidies for aluminium smelters for the transition are an “internal tariff” and anti-competitive.  Also says Net Zero is a “fantasy”.

-	Support free trade and no tariffs of any kind.

-	Says that having a Senate seat would enable him to “shine a light on the system, the deep state in Canberra” and that it’s’ more useful than 1/76 seats in most debates.  He doesn’t seem to want an overthrow of the state but rather a massive reduction in the size.

-	“Weed without the woke” attack line again!  Also claims that they have policy beyond that of Legalise Cannabis, as a one issue party.  I will note that Patten is visibly laughing through this entire section.

-	Claims he stands for “civil liberties that actually matter”, which is the least veiled attack on Patten’s pandemic-era voting record of all time, and is taken correctly by her as such.


**Fiona Patten, Legalise Cannabis**

-	Fiona Patten served as an MLA for the Australian Sex Party/Reason between 2014 and 2022, and has spent most of her career as an advocate for sex workers, but in general is a progressive (small-l) libertarian.  While pingers is a media personality and has done interviews, she is the only former parliamentarian here and therefore I am holding her to higher standards.

-	For some reason I cannot understand, she has a decent backdrop (bar the RADICAL SEX LAWS sign, which makes sense in context but would probably alienate voters), is dressed well and presents well, but decided to film in a room with only two small lamps, which with the limited range of computer webcams is always going to destroy video quality.

-	Much of what she says appears to blur the line between what Legalise Cannabis the party and Fiona Patten the candidate believe; being a one-issue party, and with her having had previous party positions, this is inevitable.  As a result, please keep in mind anything I list as a “policy” outside of cannabis is likely more what Patten believes and would do on an issue than a consistent national platform.

-	Very strong performer and if it wasn’t for the final question I would have had her at the top of the list.  Considering the anchor around her neck that is a one-issue party, she stretched cannabis to its absolute limits and also had a good deal of initiative on other issues and showed a good eye for when others had misstepped.  She also appeared, despite literally being a politician, as surprisingly human.

-	They support legalising cannabis, good to know.

-	Opens by discussing her achievements as an MLA, claiming a “track record of being an effective crossbencher”.

-	Believes that CoL is the result of global issues, but that housing costs contribute.  While legalising cannabis will help create money, she acknowledges that it won’t single-handedly solve the issue.

-	Wants greater supports for those on Jobseeker and/or the homeless, as a way to alleviate the problems.

-	Does not support the NSW Libertarian platform of reducing services as part of reducing government.  Instead, wants more money into education as it will increase productivity, and therefore make us more competitive.   

-	Also against any cuts to “women’s reproductive health”.  I put that in quotations because nobody is specific on what should be cut.

-	Against abolishing the Department of Education.  Cites the Gonski Report as to how education could be improved without removing federal oversight.

-	Identifies that rent control is a state rather than federal issue.  Is ambivalent on it at a state level.  Says it was not mentioned in the inquiries she was involved with as an MLA.

-	Wants more homes built, especially for those who are at lower socioeconomic levels.  Generally supportive of the Albanese housing agenda, though she leaves the door open to supporting “some” Greens amendments.

-	Wants the term changed from “landlord” to “housing provider”, with the associated role shift and responsibility shift.

-	The party does not have a formal view on immigration!  Patten says they will base their response on “evidence, civil liberties, compassion, social justice, personal freedoms and human rights”.  Disagrees with One Nation and the Libertarians on immigration, and calls them “cruel and callous”.

-	Believes that we cannot isolate ourselves from the world, and that it is selfish to do so. Believes we have a responsibility to continue accepting immigrants as we have done in the past. 

-	Disapproves of our current asylum seeker policy.  

-	Australia should stand up to Trump and criticise him publicly.  Wants to “tax those tech bros”.  Tells the PM to grow a pair. 

-	Is asked about Jones’s comments on the hemp industry and stops short of agreeing, but says it can be used to build houses rapidly in many places.  In my opinion it read a lot like her trying to not throw her compatriot under the bus. 

-	Supports legal, regulated cannabis so that there are quality and control standards.  Disagrees with the Libertarians on this, and believes it will create a “dangerous product”.

-	Patten says that the party may establish other issues eventually, but are currently focused on one issue.  Says that you can look to the elected Legalise Cannabis MLA’s for an idea of what will happen if you vote for them on other issues.

-	Is happy with road law changes in Victoria around cannabis, where if you have a prescription and are not impaired, you will not be fined or lose a licence.

-	Believes she made the right choice around pandemic-era legislation, based on the evidence available at the time.

**Warren Pickering, One Nation**

-	He “works in the construction industry” and you can tell.  His appearance is easily the least qualified of the four, looking like he just came home from the worksite.  The others all look like some form of politician, while he looked like just a guy at home and had a webcam to match.  Lighting was decent but no background, which makes him the only person in either debate so far to have not even tried to have a relevant backdrop.

-	Generally just less effective than his far-right contemporary, both in having political knowledge and in the charisma to appear an effective debater.  Pingers caught him out multiple times with stats that he just didn’t have a counter to, which made him look uneducated and less across the issues.  While they might have disagreed vehemently, both pingers and Dittloff were consistently able to argue each other’s points without tripping up, and both clearly had deep knowledge of the topics.

-	Unfortunately, the primary reason I have him in third instead of fourth is part of the section I cannot talk about, and even then he didn’t exploit it properly.  But for a guy who kept fighting and losing it was a remarkable turnaround.

-	I do find it funny that Puglisi asks him after several minutes of Patten, pingers and Dittloff going at it over funding cuts, “Anything you’d like to cut?”.  Pickering says that everyone agrees government is too big and too expensive, but no specifics.

-	Migration is causing “existential demand” for housing.

-	Claims 70% increase in housing companies going insolvent in Victoria over the last two years.

-	Disagrees with the immigration auction idea.

-	Claims there are 75,000 people who are here illegally and that they should be deported immediately to “make an example”.  This likely explains what Roberts was referring to the other night, but with far more detail and justification.

-	Wants Net Zero migration, which is currently 135,000 people.  Agrees with Dittloff that mass migration is being used for artificial growth, but also says it is harming communities and gives a genuinely heartfelt concern about how his daughter doesn’t have the same opportunities he did as a young adult.

-	Wants to increase the amount of time before you can apply for citizenship to 8 years.

-	Says that asylum seekers should be settling in countries near them.

-	Will not be doing burqa-style theatrics. Says that we need “real people from the real world” in the Senate and as a Senator from a larger party he’d possibly be part of the balance of power.

-	Supports “a discussion” on full legalisation.  Also thinks that we should consider legalising MDMA.

**Jordan “purple pingers” van den Lamb, Victorian Socialists**

-	The thing that stuck out to me most was a lack of preparation.  Pingers was calm and assured on camera and had a good webcam setup, and spoke clearly and eloquently.  On subjects he knows intimately, he was also excellent.  Clearly, he’d put in some effort and if he wasn’t an above-average speaker he wouldn’t be where he is now.  But it was equally clear from some of the points that he hadn’t read his opponents platforms, such as when he was surprised the Libertarians oppose AUKUS (while this is not directly stated on their policy platform, they are against any foreign deployments).  Both Rennick and Dittloff showed that preparation and knowing your mark are so, so important in these debates and it just feels like a missed opportunity for him to attack one of the far-right parties.

-	He also struggled to escape the mold of a university socialist in talking and manner.  His way of speaking felt familiar to anyone who has ever dealt with SAlt on campus, with the phrasing and words used to make his arguments guaranteed to alienate anyone not already sympathetic, and his appearance and background were very stereotypical.  Even at the end, when Patten asked if everyone would agree to drinks (a softball question to humanise and bring together everyone) he was awkward while everyone else read the room and agreed (or in Puglisi’s case, made the “I’m too young” joke).

-	Many of his arguments felt as if they were to be used with people who already agree, rather than the undecided; Patten was far better at showing just why a policy would negatively affect regular people, rather than just stating it and assuming everyone would agree.  Considering a good chunk of the audience weren’t your typical purple pingers viewers, I really do think there was a missed opportunity, especially on the forbidden topic and any time Dittloff brought up cutting something that the average person would likely support keeping.  

-	When he did have arguments, I will say that unfortunately the stream kept cutting out on 6News’s end pretty frequently, which of course isn’t his fault and means that he may well have had much stronger arguments that we just didn’t hear.

-	Came straight out the gate by saying that the housing crisis is caused by capitalism and then again for the cost of living crisis not a minute later.  Wants “organising in our streets, workplaces and unions”.  

-	Wants to cut “the tens of billions of dollars to subsidize landlords”.  Very strong against Dittloff on the NDIS, where he does not support cuts.

-	The Victorian Socialists/pingers housing platform is far more reasonable than you might expect from his demeanour, as it’s a slightly amped-up version of the Greens platform.  Unfortunately I had to look it up because (and this isn’t his fault) the stream died on 6News’s end before we got into it.  When the stream returned he was talking about rent seeking.
-	Believes that building more housing will not improve quality (due to capitalism incentivising sub-standard construction).

-	Migration isn’t causing the housing crisis, a lack of government intervention is causing it.  Wants a public builder and rent controls (5 years per his website, then CPI).

-	Doesn’t like AUKUS, B-52’s in Darwin and a casual “if they exist” surrounding the potential of Australia and the US having mutual enemies.

-	Will represent the working class in parliament, but there’s a strong undertone that any meaningful change will be extraparlimentary and outside democratic norms.  The fact that he wants to **overthrow the government** is dropped as a casual, indirect aside an hour into a debate.  Easily the most extreme thing said by anyone at any point. 
 
-	Will take the median wage if elected, and use the rest to “amplify community organising” as well as help striking workers.

-	Wants legalisation but believes that there are more important issues on the table.
**Post 3:**

> IVO's quantum drive: will it transform the future of space technology or is it just orbital snake oil? TLDR: It’s snake oil

* The creator of the quantum drive has no formal education in physics or engineering.
* He is a pastor at a congregation associated with anti-science beliefs and other extremely harmful ideologies.
* He claims the quantum drive doesn’t break the laws of physics, rather, the laws of physics are incorrectly understood by most academics.
* While he claims various ground tests have been performed (including vacuum testing) and that thrust was achieved, no data has been released from the tests.
* No papers detailing IVO’s specific drive design have been peer-reviewed or released in pre-print journals or any other form.
* Since no technical information is available, information about the drive’s creator and other IVO staff is discussed as a means of assessing legitimacy of claims.

In the last few months, IVO limited has been generating a lot of buzz on reddit and elsewhere around their payload on the BARRY-1 satellite. Like many satellites before theirs, IVO's payload was launched into orbit back in November 2023, ride sharing on one of several small satellites being sent into space in that launch. However, they claim that their payload on the satellite consists of two engines unlike anything the world has ever known and that they have the potential to fundamentally reshape everything we know about physics. IVO claims that these are "Quantum Drives." The design of these propellantless engines breaks the established laws of physics as most academics understand them.

IVO's founder, Richard Mansell, who is also the creator of IVO's Quantum Drive, circumvents this argument, in favor of a more nuanced one. Instead of claiming to break the laws of physics, he argues that the laws of physics are incorrectly interpreted by most people. Essentially, he argues that his technology doesn't defy the laws of physics, rather, most of the scientific community doesn't understand the laws of physics. However, he does, and his technology is supported by the (disputed) theories of Mike McCulloch, a physicist with extensive writing on a phenomenon he proposed called quantized inertia theory.

Mansell has generated a bit of buzz as of late, posting pre-publication journal articles claiming that his tech has robustly passed a substantial campaign of ground testing. Per their claims, the engine has consistently been found to generate thrust, and alternative explanations like measurement error or other artifacts have supposedly been ruled out.

It would be easy to write this off as yet another scam, or to relegate it to the list of other technologies like the EmDrive that made similar claims and showed promise but never materialized into anything more. Except that IVO has put their money where their mouth is. They reportedly have two prototype engines in orbit, and they plan to test them soon. Since arriving in orbit and passing initial systems checks, the timeline for start of testing has slipped a couple of times. Still, IVO claims the tests will happen soon.

If successful, the implications are huge. A foundational shift in our understanding of physics will be required. But is that likely? Or will this all turn out to be just another episode of grifters in space?

Since there's limited information available about the technology itself (and most of us honestly wouldn't be qualified to interpret the technical data even if it was publicly available) it may be more helpful to examine the information that is available regarding the two people in charge of the project. IVO is a technology company. Neither their CEO (the inventor of the quantum drive) nor their COO have degrees in engineering or physics. Many great minds have accomplished a great deal without formal instruction in the sciences, so this certainly doesn't rule out the possibility that the IVO team has accomplished what they've claimed.

IVO's CEO is Richard Mansell. Details regarding his education aren't provided on his LinkedIn page, but I found an article in the July 2021 edition of local news outlet the Valley Business Front in which he discusses his science background:

"Without an electrical engineering degree Mansell says, “Praise the Lord,” he went about experimenting without being encumbered by the “rules” regarding electrical power transmission. People have discounted his concept, industry leaders he says, even “dismissed” it outright in the past. It ultimately led to his now patent-pending technology, the ability to use those non-emf electrical fields, which he says, “are natural,” and all around us."

IVO's website lists a lot of accomplishments Richard has made, many of which are in relation to collaborations he had in the past with a now defunct auto company called Local Motors. Richard and the man who would later become COO for IVO, Daniel Telehey, reportedly met one another in the online community forum for Local Motors back in 2014. The company had a unique model where they actually crowdsourced design of vehicles and had buyers come in to company owned micro-factories to build the vehicles themselves with the help of employees on the premises. They made the Rally Fighter, a pretty epic car that was both highly offroad capable and street legal. Credit where credit is due, it was a cool vehicle. I'm not sure how much input Richard had in the design, but I've seen vague mentions (mostly by himself or by IVO's COO) of contributions on the software and hardware side.

Flipping through the IVO website, I also found another mention of IVO's collaboration on another vehicle with Local Motors, the infamous Olli 2.0 autonomous shuttle. In the article, they discuss partnering with the company to provide hardware for Olli 2.0. I'm not so sure that's something I'd be boasting about, given that in 2021 the pilot program testing the shuttle was shut down after a only a few months in operation after the vehicle crashed into a tree and critically injured the onboard safety attendant. Mind you, this is a vehicle with a top speed of 25 mph. Local Motors closed down a month after the accident.

I don't know this man. I have very little information about him, who he truly his, and what he is capable of. So, my impression of whether his claims are true has to be based mostly on conjecture. Let me clearly acknowledge that I am aware that my own personal biases obviously shape the assumptions involved in my statements in this post; there's no need to come for me in the comments.

Clearly, he's done some engineering work. He's self-taught and seems to have developed some patents on his own. From the looks of things, he's likely gifted, at least in terms of his mechanical abilities. But you can be gifted intellectually in one area and be woefully ignorant in another.

In asking the question, "is this person capable of the sort of thinking that challenges the entire foundation of physics as we know it?", I think its fair to look at publicly available information that illustrates that person's judgement. While this next bit approaches the subject tangentially, bear with me, as it seems relevant.

Richard is the pastor at a church in Virginia. I believe everyone has a right to their personal beliefs, and I won't criticize anyone for believing in a god or some other higher power. However, there are limits to that. When someone uses their power or their faith in ways that directly cause harm to others, or in ways that maintain oppressive systems with a high potential for harming others, it becomes ok to challenge those beliefs. As such, the people he has chosen to surround himself with and who he keeps in positions of power in his church speaks a great deal to his judgement. I think there’s inherent value to looking at this sort of thing as we seek information on whether we should be inherently skeptical/suspicious of someone.

In addition to being generally anti-science, the leaders of the vacation bible school at Pastor Mansell’s hold some particularly deleterious beliefs. As the head of his church, I have to assume that Richard agrees with the beliefs of the couple he has charged with running his bible school. Unfortunately, they adhere to a version of Christianity that is, at its best, factually incorrect, and, at its worst, abusive and harmful to an extent that I find to be both grotesque and morally repulsive.

Before you judge me for judging them, read a few more paragraphs. I grew up in an extremely conservative religious household. I'm not shocked by the typical anti lgbtq+ rhetoric so many churches espouse. Saddened by it, yes; shocked, no. But, in addition to your standard, run of the mill hateful rhetoric against the lgbtq+ community, Pastor Mansell's team members proudly endorse bigoted and harmful ideologies that somehow manage to take things farther in ways that managed to shock even my rather indelicate sensibilities. The people Pastor Mansell has appointed to watch over and teach his congregation's children, to safeguard them and shield their minds and spirits, explicitly state that divorce is never acceptable, even in cases where one spouse has engaged in sexual, physical, and/or emotional abuse of a child or spouse (or any other crime for that matter). The following quotes come directly from the website of their organization, Creation Family Ministries (https://creationfamilyministries.org/marriage/).

"What about divorce? What about abusive situations or incest? These are certainly valid concerns and important questions. However, they must all be considered within the framework of God’s design for men and women. Certainly there are times in which marriages are devastated by sinful behavior. But does wrong behavior necessitate additional wrong actions such as a separation with a goal of divorce? “For all have sinned…” (Romans 3:23). The bringing together of a man and a woman who have separated from their parents is a design by the Creator to be a life-long commitment. This is NOT to say there won’t be hard times or difficult circumstances. Doesn’t the familiar wedding vow say “‘Til death dho \[sic\] us part” or “in sickness and in health…?"…

"Are there extenuating circumstances that strain marriages? Of course! Does it mean that divorce is necessary? No! In fact, divorce is never necessary. God is a life-changing God, and it may be that we want to get out from under a situation before we learn the lesson God has for us. Perhaps it was a situation that was a bad choice on our part in the first place. Sometimes, bad situations are not our fault or of our choosing. However, God has something for us to learn through “all things.”

You read that correctly: they refer to abuse and incest (i.e., the rape or other sexual assault of one's own children) as "extenuating circumstances" and suggest it would be incorrect to "want to get out from under a situation before we learn the lesson God has for us." In these peoples' eyes, even in chronically abusive situations, there's no excuse for a divorce and seeking one would be evil…

Shame on these people, and shame on Richard for allowing these people to operate in a position of power in his church. I own my bias here, and I stand behind it: any person who chooses to keep these people by his side as leaders in his congregation is far too backwards to think in a manner that could possibly lead to a redefinition of the laws of physics.

Let me further highlight the dark absurdity of the situation by shining the light on a though Richard felt was important enough to bring attention to on his social media. In a tweet from 2021, Pastor Richard shills for a company that lifters offensive content from media: “If u/Disney really was concerned about filtering out the offensive in their shows, they would have licensed with u/VidAngel”.  Personally, I'd be a lot more worried about removing the plank from my own eye (i.e., the abusive systems my church has enabled that will force children and spouses to remain trapped in unimaginably horrific cycles of abuse, resulting in lifelong scars that will never fully heal).

I should also probably note that, surprise, they also don't believe in evolution and have their own curriculum of anti-science indoctrination that they teach the kiddos. While they don't believe in evolution they do believe in sclerology, a branch of pseudoscience founded on the belief that visual examination of the sclera (i.e., the whites of the eyes) can reveal up to date health information about an individual and that requires no clinical tools or other medical testing. Conveniently, they also hawk related herbal supplements to cure anything that ails you. Grifters do like to associate  with one another.

Moving on. IVO’s COO, Daniel Telehey, has an associate’s degree in Power Plant Technology from Bismarck State College. The degree requirements don't include any science or physics related coursework (https://bismarckstate.edu/academics/programs/PowerPlantTechnology/). Not that a COO needs that, but since he and the CEO seem to be about 66% of the team at IVO, it seems worth mentioning. His LinkedIn does state that he worked as Chief Project Developer for a company called Waco Motors from Oct 2014 to April 2017. Which is odd, because the LLC documentation for Waco Motors shows the company was formed in June 2015 before being involuntarily dissolved in 2017. His writeup on the IVO website states that, shortly after leaving the military, he founded Waco Motors, where his biggest achievement was purchasing the rights to produce the Rally Fighter, the previously mentioned car designed by Local Motors. Thing is, Local Motors remained open and continued to produce Rally Fighters until January of 2022, suggesting he likely couldn't have owned the production rights. Their whole marketing model was selling the parts to individuals so that they could independently build their own vehicle. I'm assuming that's what he did and he's trying to misrepresent himself as buying the full rights to the intellectual property. Could be wrong, but seems more plausible to me.

What's my goal in sharing all of this? Namely, to give the reader an opportunity to access more information (some from firsthand sources and some that's admittedly based on conjecture) for consideration when making their own determination about the legitimacy of IVO's claims. Now, of course, none of the things discussed above would prevent Richard and Dan from developing a device that requires the laws of physics to be rewritten. But having taking a slightly deeper dive into their backgrounds, I’m certainly far more skeptical about their claims...

My personal predictions regarding the two most likely possibilities for what we'll see with IVO's tests of the supposed quantum engine on BARRY-1:

1. It won't do anything because the science is flawed.
2. The elevation of the orbit will increase due to thrust produced by IVO's engine, however, it won't be because the tech they have on BARRY-1 is what it's been claimed to be. Rather, future attempts to replicate their findings by outside teams will fail and it will later be revealed that IVO's payload was in actuality a conventional engine that used already established technology to achieve their purported success.
**Post 4:**

> Australia is in an extinction crisis – why isn’t it an issue at this election? | Endangered species **Some of the country’s most loved native species, including the koala and the hairy-nosed wombat, are on the brink. Is this their last chance at survival?**

**Adam Morton, Mon 7 Apr 2025 01.00 AEST**

Most parliamentarians might be surprised to learn it, but Australians care about nature. Late last year, the not-for-profit Biodiversity Council commissioned a survey of 3,500 Australians – three times the size of the oft-cited Newspoll and representative of the entire population – to gauge what they thought about the environment. The results tell a striking story at odds with the prevailing political and media debate.

A vast majority of people – 96% – said more action was needed to look after Australia’s natural environment. Nearly two-thirds were between moderately and extremely concerned about the loss of plants and animals around where they live.

Unsurprisingly, the cost of living was way ahead when people were asked to nominate the issues they would like leaders to prioritise. But the environment was in a peloton of four issues vying for second place, alongside housing, healthcare and the economy.

On what they would like to see done, three-quarters of respondents said they would back stronger national nature laws, including the introduction of clear environment standards against which development proposals could be measured and potentially rejected.

Any survey should be treated cautiously, but the Biodiversity Council’s director, James Trezise, says it is not a one-off – the results are consistent with the findings of similar surveys in 2022 and 2023.

They are also clearly at odds with where Anthony Albanese ended this term of parliament, with Peter Dutton’s support. After bowing to [an aggressive industry-led backlash in Western Australia](https://www.theguardian.com/australia-news/2024/nov/27/albanese-kills-off-deal-with-greens-to-pass-nature-positive-law-after-intervention-by-wa-premier) to shelve a commitment to create a national Environment Protection Agency, the prime minister rushed through a law [to protect Tasmanian salmon farming](https://www.theguardian.com/environment/2025/mar/26/controversial-bill-to-protect-tasmanian-salmon-industry-passes-despite-environmental-concerns) from an environmental review.

Longtime campaigners say it meant the term began with a government promising the environment would be “[back on the priority list](https://www.theguardian.com/environment/2022/jul/19/labor-says-it-wont-put-head-in-the-sand-as-it-releases-shocking-environment-report)”, including a once-in-a-generation revamp of nature laws, but finished with existing legislation being weakened in a way that could yet have broader ramifications.

The message from peer-reviewed science is blunt: Australia is in an extinction crisis.

Over the past decade, more than 550 Australian species have been either newly recognised as at risk of extinction or moved a step closer to being erased from the planet.

The[ full list of threatened Australian animals, plants and ecological communities](https://www.environment.gov.au/cgi-bin/sprat/public/publicthreatenedlist.pl) now has more than 2,200 entries. It includes some of the country’s most loved native species, including the koala, the Tasmanian devil, the northern hairy-nosed wombat and a range of the type of animals that Australians take for granted: parrots, cockatoos, finches, quolls, gliders, wallabies, frogs, snakes and fish. Scientists say that, unless something is done to improve their plight, many could become extinct this century.

Partly, this is linked to a global threat – what is described as the world’s sixth mass extinction, and the first driven by humans. But part of it is specifically Australian and avoidable.

A 2021 government state-of-the-environment report found the country’s environment was in [poor and deteriorating health](https://www.theguardian.com/environment/2022/jul/19/labor-says-it-wont-put-head-in-the-sand-as-it-releases-shocking-environment-report) due to a list of pressures – habitat loss, invasive species, pollution and mining, and the climate crisis. Australia tops global rankings for mammal extinction – at least 33 species have died out since European invasion and colonisation – and is number two behind Indonesia for loss of biodiversity.

In recent weeks, the overriding question from scientists and conservationists who dedicate their lives to protecting the country’s unique wildlife has been: what will it take for national leaders to take the issue seriously? And will this campaign – and the next parliament – be a last chance to hope for something better?

“A lot of people in the scientific and conservation community have found the last three years exceptionally frustrating. A lot was promised, but in the end we went backwards,” Trezise says. “The survey shows there is a clear mismatch between what Australians expect the government to be doing and what it is actually doing. And it found there has been a decline in trust in politicians on the issue, particularly the major parties.”

Over the next week, Guardian Australia’s environment team will tell the stories of passionate people trying to circumvent this in their own quiet way by working to save threatened animals.

This work is most often done with little, if any, government support. One of the findings from the Biodiversity Council is the extent to which Australians overestimate the federal government’s commitment to biodiversity. Most guess that about 1% of the budget is dedicated to nature programs, a proportion that the [Greens have said they would argue for](https://www.theguardian.com/australia-news/2025/apr/02/australian-election-2025-greens-albanese-federal-epa-environment-adam-bandt) from the crossbench, and that would translate to about $7.8bn a year.

Trezise says that, while funding for nature has increased since Labor was elected in 2022, the reality is that in last month’s budget, on-ground biodiversity programs received just 0.06% of spending – or just six cents for every $100 committed.

Lesley Hughes, professor emerita at Macquarie University and a senior figure in environmental science as a member of the Wentworth Group of Concerned Scientists and the Biodiversity Council, says it is a “deeply depressing” figure given most people would think even 1% was a “pathetically small amount” to save species from extinction and preserve places they care about. “I do think it shows politicians totally underestimate how much people care about nature,” she says.

She says it is tied to a broader lack of understanding that a healthy biosphere is “our life support system”. “We should treat it as a precious heritage item that is irreplaceable, and we need to see ourselves as part of nature. We are just another species,” she says. “OK, we’re a clever, resilient and adaptable species, but we’ve destroyed so much because we haven’t seen ourselves as being dependent and a part of it.”

The failure to directly address entrenched issues in environmental protection is not new. The singer and activist [Peter Garrett](https://www.theguardian.com/australia-news/peter-garrett) had first-hand experience of how nature is considered in government decision-making, having served as Labor environment minister between 2007 and 2010. Garrett blocked development proposals more than other environment ministers, but says nature protection was rarely seen as a first-order issue by leaders in government and bureaucracy. He has seen no substantial improvement since leaving parliament.

“That’s a tragedy, particularly given the policy commitments that the current government had when it came into power,” the former Midnight Oil singer says.

“The problem that we have is that, whether it’s at a federal or at a state level – and notwithstanding the best intentions and efforts of environment ministers and \[non-government organisations\] and scientists who advocate on behalf of nature – when it comes to the final decisions that are taken in the cabinet room by political leaders through the prism of economics, nature always comes last. There are exceptions to that, but very few.”

Garrett says addressing that requires a shift in thinking at the top of government, but also across the community. He agrees Australians love nature, but says it often becomes a lower-order issue at the ballot box. It means that while conservation gains are possible – for example, an [expansion in protected areas](https://www.tanyaplibersek.com/media/media-releases/labor-will-save-australia-s-iconic-bushland/) and support for First Nations ranger programs under Labor in this term – they mostly happen in places that no one wants to exploit.

“It’s very difficult in this country to break the cognitive dissonance between us loving our wildlife and enjoying an incredible environment and actually putting resolute steps in place to make sure that it’s protected, even if that comes at a cost,” Garrett says.

Trezise says that is backed up by another finding from the survey – that while people care about nature and want more done to protect it, they have little real insight into how steep the decline has become, or what a biodiversity crisis actually means.

Part of what it means goes beyond what is captured by a threatened species list. It also refers to the loss of diversity within species and ecosystems, including local extinctions of once abundant creatures.

This has become a common story for many Australians who have watched the disappearance of wildlife from particular areas during their lifetimes. To give one example: Brendan Sydes, the national biodiversity policy adviser with the Australian Conservation Foundation, lives in central Victoria. [Grey-crowned babblers](https://www.birdsinbackyards.net/species/Pomatostomus-temporalis), birds with a curved beak and distinctive cry that are found across tropical and subtropical areas, were common in nearby bush earlier this century, but in more recent years have vanished.

“For us there’s a sort of a continuing discussion of: have you heard these birds recently? And the answer is: maybe they’ve gone, maybe we’re not going to see them any more. And that’s the legacy of fragmentation of habitat and the vulnerability that results from that,” he says. “The same thing is happening with other once common species. And once something’s gone from the area, it’s likely gone forever.”

Sydes says it is easy to become immune to this sort of decline. “It’s become a sort of feature of Australian nature. We really need strong, dedicated action for it to start to turn around,” he says.

How to respond to this is a deeply challenging question. It could start with an end to the government greenlighting the clearing of forest and woodlands relied on by threatened species. The Australian Conservation Foundation found nearly 26,000 hectares – an area more than 90 times the size of the Sydney CBD – was [approved for destruction last year](https://www.theguardian.com/australia-news/2025/jan/21/threatened-species-habitat-destruction-labor). Under the existing laws, far more clearing than this happens without federal oversight.

The challenge is not only to stop the loss of habitat but to restore the environment in places it has been lost in 250 years of European-driven clearing. Experts say that becomes particularly important in an age of climate crisis, when species adapted to living at particular temperatures and with particular levels of rain are being driven from their longtime habitats as the local conditions change. Connecting fragmented forests and other parcels of nature will become increasingly important.

Laying over the top of this is the impact of invasive species that kill and diminish native species, but are now so pervasive that they have changed the landscape for ever. It means there is no going back to the environment of 1750. A question that the political debate over the environment has yet to fully grapple with is what success from here actually looks like.

The environment minister, Tanya Plibersek, raised the idea this term by promising a “nature positive” future, adopting a term that overseas has been defined as [halting and reversing nature loss by 2030 measured against a 2020 baseline](https://f.hubspotusercontent20.net/hubfs/4783129/Nature%20Positive%20The%20Global%20Goal%20for%20Nature%20paper.pdf) and achieving “full recovery” by 2050. It would require retaining existing natural ecosystems – both areas that are highly intact and remnant fragments – and starting immediate restoration work on damaged and lost nature areas.

But achieving that will demand significant funding – whether from public or private sources – in addition to tightening laws to prevent further destruction. Labor passed legislation to encourage private investment, but hasn’t explained how, or when, it will arrive in significant sums.

It is unclear if the “nature positive” tag will survive into the next term given it has been rejected by WA industry, among others. Albanese has again promised to fix the laws [and introduce a different model of EPA](https://www.theguardian.com/environment/2025/mar/31/albanese-says-federal-epa-will-not-be-same-model-as-earlier-one-he-promised-but-didnt-deliver) to that promised this term, but given no details. Dutton says no one can say the existing [environment protection system is inadequate](https://www.peterdutton.com.au/leader-of-the-opposition-transcript-address-to-minerals-week-2024-conference-canberra/) and promises faster decisions to allow developments to go ahead.

In her darker moments, Hughes wonders if anything can change, but she says she remains an optimist. She sees signs of a resurgence in the idea that people need to connect with and value nature as important to the human race, and says it could make nature and species conversation become a higher priority. “Let’s hope that’s the case,” she says.

Garrett says the path ahead for people who want change needs to be “building community and organisational strengths”, and supporting activists prepared to put themselves on the frontline using nonviolent direct action against fossil fuel exploitation and environment destruction.

“It’s about a transformative ethic that lifts what we have and recognises what we have been able to secure jointly,” he says. “It gave us a great conservation estate – look at the world heritage areas, look at Kakadu – but those great gains are in danger.

“Are we going to see unfettered housing developments and oil and gas exploration basically take over every square metre of the continent that’s left for them to do it? Or are we going to draw a line in the sand? It’s time to draw that line.”
**Post 5:**

> A Comprehensive List of Restrictions and lack of rights that youth face. I am gathering such a list for a text on Youth Right's that i am writing. So Heya Guys, it has been a while since i last came onto this server. But such are my illnesses, and all instances of psychiatric illnesses, that I do not have the strength to do my wishes.

Recently, upon being dogpiled on a Subreddit for "writing fanfiction glorifying the rape of children" ( What i was talking about was absolutely unrelated to that, someone dug into my reddit history, misquoted me on a youth right's topic and used it to dogpile me), I have had an incredible reinvigoration of my wrath. That wrath is manifested in this text I'm writing for Youth Right's, which i'd gladly love any help on. But part of it, is to address and raise those topics about youth right's no one talks about. Not just the usual lack of voting rights, unable to drink and have peace,, can't enter into sexual relationships or else it's "Statutory rape" ( How this is rape, Only God knows) but other restrictions us Youth's face such as Not being able to participate in clinical trials and biomedical research, donate blood and publish research papers, leading illnesses like psychiatric, neurolgical etc illnesses in youth to basically have less information than on 80,000 BC humans, and also our restriction on publishing Articles and pieces in newspapers and media to voice our views on any topic, Unable to publish our novels, fantasy and sci fi and speculative fiction to be precise or publish art ( The latter is easier since there's no intermediary like a Publishing agency. But good luck getting commissions for your art lol), and this last point and the first one hit particularly hard for me, as an author and artist who has many psychiatric illnesses but couldn't take part in research and clinicial trials or donate blood or contribute to research papers before age 18)

I want to make a comprehensive, exhaustive list of Such evils. Of course, given that we face so many restrictions that we lose count and can't name them all and there's so many restrictions, we have to up age ourselves on social media. I'd like points ( I swear to god i had so many beautiful points, like restrictions on publishing novels and art, writing pieces in newspapers and media and participating in clinical trials and biomedical research and donating blood and contributing to research papers and also Us not being able to even write about or see sex, for example few days ago, i was hearing stuff about how if a ship ( A fictional relationship) between two characters from the Game Genshin Impact was 'legal" or not, in the holy name of God, did it burst my brain).

The following are restrictions that i have gathered about and could "collect", some of them due to my own experiences.

1) Restrictions on Participating in Clinical trials, Biomedical research, contributing to research papers and donating blood- What does this lead to? I talk about more in detail in my text which is still a work in progress, but it means that "old people illnesses' like psychotic illnesses, or Dementia like illnesses ( Which i have due to long covid, both of them) are not seen and treated in youth, even though most of my friends with psychotic illnesses had theirs began around the same time as mine, age 15. I have so much more to say on this and even the previous point, i elaborate on it a lot because we deserve our illnesses and pains to be heard.

  
Number Two ( The two key on my laptop is broken so i can't tyoe it- If any of us dares to do worldbuilding, write fantasy and speculative fiction novels and wants to get it published, well, we can't because publishing houses won't allow a "minor" to "sign contracts". Same thing with making your own art and making money off it, or making your own anime and animations and any creative works. This one particularly hurts me because i'm brimming with worldbuilding and novels in my brain, and have even written many of them down ( Alas, my illnesses have destroyed all my worldbuilding capabilities and writing capabilities). We can't express ourselves artistically and i cannot explain just how much this hurts, because I personally want my novels out there and people to honour them. But of course, I'm too mature for my own novels since i write about NSFW topics like Psychiatric illnesses, war, sex, violence etc.

  
3) Restrictions on Publishing Articles and Pieces in Newspapers and other Media- The media is perhaps the only refuge of the oppressed ( It makes sense since i live in a country that not even two months ago, overthrew a totalitarian police state) and where we can express our views, where the common person has power. But we cannot write in most newspapers which don't allow "Minors" to publish articles. So we can't ever speak our views, on whatever topic they may be.

  
4) Restrictions on Joining Organizations and etc- Once upon a time, i had joined the staff for a Magazine about Worldbuilding. Many thanks to the creators for allowing me into the team when i was a minor but telling this to another friend of mine, he was surprised that they "allowed a minor" into their team. Like WHAT THE FUCK? WHAT DOES MY AGE HAVE TO DO WITH IT? Then again, i remember an accursed statement given by my older brother ( In the context of Under 18's having sexual relationships with those older than them, that Minors cannot consent. Like, congrats on admitting that youth are treated as literal robots, without any power to consent.). And whatever organizations we can join, it's just useless volunteering. Direct Action ( I am so glad that in the recent revolution, mainly students and people from the age range of 14-30 lead the revolution, as incomplete and scrambled as it was, but that's another arguement) and revolution? Hell, you can't even join various fandoms because they are 18+ only. Asking for allowing us to direct action such as say, supporting rebel groups fighting against dictatorships such as in Myanmar is a faraway dream. Most youth organizations are useless. The moment you advocate for radical change like abolishing and eradicating this entire adultist and ageist ( Towards both the Youth and the Aged, i have felt this a lot, Elderly people are treated like children, just like us, so it's why i've often had solidarity with some old people. But then again, many times, a lifetime of adultism and prejudice towards youth transforms them into the demons they are now. Thankfully, my parents and grandparents weren't like this) system and every last oppressive system like overhauling the healthcare, particularly psychiatric healthcare system so that undertreatment, undermedication and downplaying of people's illnesses ( Like how it happened to me, when my 4th psychiatrist, may his accursed name not be taken even by mistake, told me "Why is a young man like him on so many medicines"? And took me off two medicines i was responding to wonderfully, and didn't replace one of them, leading to 10 days of hell for me) or the social justice and welfare system, then we get shut down. And we get particularly shut down because we're rebellious and ironically, contrary to the demonic myth ( It is demonic and a myth) that "hormonal teens" make impulsive decisions, We not only are not impulsive but very wise too, oftentimes having greater wisdom than 40 year old's who have not an iota of information about how the world works and should work. 

  
5) Already, we are tormented and face god know's what if we happen to fall in love with someone older than us, to the point we want to share our bodies with them. But God forbid, we will be "traumatized by sex scenes, war, blood, gore, death and rape" and here i am, who seeing sex scenes, feels at peace ( Sex scenes that involve love and healing, like the Sex Scene from Fate, Heaven's Feel Anime movies, or from my own novels and works). I shouldn't even need to explain this topic. We cannot even enjoy certain works of art. Why? Because they have NSFW content which apparently us "Minors" can't handle and will be traumatized by ( Maybe they should understand that their restrictions are what's traumatizing us? And no, this isn't love. Paternalism isn't love, and i will perish on this hill. My Father allowed me to play violent video games, watch sex scenes ( I joke not, from age 11, i watched and read sex scenes and whenever my father walked in, he was so casual even though i was mortified, and he has always protected me in a way Anglo American parent's ever could not. My Father always answered my questions on NSFW stuff and supported my Youth right's views ever since i had them. He even said himself that if i wanted to get married with someone 10 years older than me when i was 14, he said he'd happily do it ( His only concern and opposition was wondering if anyone would even agree to marry a 14 year old because we lack employment and are "immature" and "inexperienced" and don't have the "life qualities for marriage", but I know that he's just talking about how no one would agree to enter into a committed relationship with me at age 14 ( I am 19 years of age now). So Maybe they should be like my father, whom i love beyond words.

6) On alcohol and Drinking- Endless "studies" have been done saying how alcohol harms the "teenage developing brain" and shouldn't be taken until age twenty five ( Seriously?) and well, what do i say?

If i had to give a rant on alcohol, It'd take up an entire novel. All i want to say is that teen's don't turn into addicts. People in my village have been drinking and fuck, taking opium with it ( Something most Anglo American's would faint over, because we're taking "DRUGS!") and guess what? In two hundred years, my family has failed to produce a single addict. It's mysterious isn't it, how the US and UK have such high rates of substance abuse with alcohol but India has so little despite us drinking more and from an earlier age. Maybe it's the capitalism and the absolute cursed nature of those lands, households and families that make them more susceptible to addiction? 

But we can't even wind down and enjoy some peace with a glass of cold Beer or wine. Because alcohol is "a deadly poison, a killer, destroyer of families and so much more" ( Meanwhile for me, alcohol has united our family lol. And so many others. Me and my parents drinking together has contributed to a lot less fights and arguements between us as of recent, after we managed to get my conservative mother into drinking). And maybe they should understand that Sugar works on the same pathways as alcohol in the brain, thus is sugar bad? No. And alcohol does not cause brain damage, and i repeat, alcohol is not worse than meth even in moderation. If we're talking moderation, alcohol is even safer than smoking, there you go. And benefits?

In this world where day and night we have to work like donkeys and sleep and even the word sedation and CNS depression is demonized, alcohol would be a nice break. It makes you feel better, happier, peaceful, serene etc. Helps me in journalling about my illnesses too. Hell, helps me even connect and empathize with others. And i'm not the only one. Alcohol really just brings out who you are truly, it does not cause a truly gentle person to go violent. 

  
7) Our voices are not heard, we are always referred to as Someone's "Teens". Like people saying "My teen", specifically indicating their age. In this case, would saying my Son/Daughter or Offspring be so difficult? And whenever we speak on social media, we aren't even heard. After all, god knows how many accounts have "Minors Do not Interact" as part of their policy ( And they're proud of it. I will go to an extreme but just length and say that this one act of theirs wipes out most if not all of their good deeds, just as certain evils like being a sadist wipes away all your good deeds like charitable actions, devotion to someone or something etc.

8) How can i forget "delinquency" laws? It existed, or still exists in my country in the most hideous form, where youth were particularly targeted by the security forces of the regime because we were rebellious. But across even "democratic" countries ( Democracy is not just about votes, it includes human and civil rights too. And us youth's don't even have the first), if you stand up to your parents or escape from abusive households, you will be handed straight back in.

9) We literally have to up age ourselves on social media and the internet because that's how all pervasive restrictions on us are. I had more in my brain like the first four points but i forgot, i wish i can get them back. And this is why i'm making this post after all.

10) No Representation in Student's bodies until we come of a certain age and constantly getting bullied by the seniors ( Happened in my school, and we didn't even have a student body). We also were restricted from going to the library. I could make an entire novel on youth right's issues related to education.



Many points like the first two came across me because they happened to me, and as they say, experience is the best teacher. I want to draw eyes on topics like those that don't ever get discussed in even the most fringe of youths rights discussions ( A hyperbole but you know the point i'm trying to get across). And I want points like that, and i want to cover a comprehensive list of restrictions we youth's face. So that we can first identify and show the world just how much and in how many ways we are shackled.  


I truly believe that other's can talk about restrictions or points that didn't ever cross me. It's not like that if they didn't cross mine, they wouldn't cross other's. After all, isn't that why we ask for help?

English isn't my first language, so pardon any mistakes i made. I would absolutely love everyone's contribution to this, for i ultimately just want to compile a list of evils and restrictions we suffer so that we can show the world, for a beginning, just how shackled we are.
## Topic 3: Digital Identity & Youth Voice
**Post 1:**

> According to all known laws of aviation, there is no way a bee should be able to fly. Its wings are too small to get its fat little body off the ground. The bee, of course, flies anyway because bees don't care what humans think is impossible. Yellow, black. Yellow, black. Yellow, black. Yellow, black. Ooh, black and yellow! Let's shake it up a little. Barry! Breakfast is ready! Coming! Hang on a second. Hello? - Barry? - Adam? - Can you believe this is happening? - I can't. I'll pick you up. Looking sharp. Use the stairs. Your father paid good money for those. Sorry. I'm excited. Here's the graduate. We're very proud of you, son. A perfect report card, all B's. Very proud. Ma! I got a thing going here. - You got lint on your fuzz. - Ow! That's me! - Wave to us! We'll be in row 118,000. - Bye! Barry, I told you, stop flying in the house! - Hey, Adam. - Hey, Barry. - Is that fuzz gel? - A little. Special day, graduation. Never thought I'd make it. Three days grade school, three days high school. Those were awkward. Three days college. I'm glad I took a day and hitchhiked around the hive. You did come back different. - Hi, Barry. - Artie, growing a mustache? Looks good. - Hear about Frankie? - Yeah. - You going to the funeral? - No, I'm not going. Everybody knows, sting someone, you die. Don't waste it on a squirrel. Such a hothead. I guess he could have just gotten out of the way. I love this incorporating an amusement park into our day. That's why we don't need vacations. Boy, quite a bit of pomp... under the circumstances. - Well, Adam, today we are men. - We are! - Bee-men. - Amen! Hallelujah! Students, faculty, distinguished bees, please welcome Dean Buzzwell. Welcome, New Hive City graduating class of... ...9:15. That concludes our ceremonies. And begins your career at Honex Industries! Will we pick our job today? I heard it's just orientation. Heads up! Here we go. Keep your hands and antennas inside the tram at all times. - Wonder what it'll be like? - A little scary. Welcome to Honex, a division of Honesco and a part of the Hexagon Group. This is it! Wow. Wow. We know that you, as a bee, have worked your whole life to get to the point where you can work for your whole life. Honey begins when our valiant Pollen Jocks bring the nectar to the hive. Our top-secret formula is automatically color-corrected, scent-adjusted and bubble-contoured into this soothing sweet syrup with its distinctive golden glow you know as... Honey! - That girl was hot. - She's my cousin! - She is? - Yes, we're all cousins. - Right. You're right. - At Honex, we constantly strive to improve every aspect of bee existence. These bees are stress-testing a new helmet technology. - What do you think he makes? - Not enough. Here we have our latest advancement, the Krelman. - What does that do? - Catches that little strand of honey that hangs after you pour it. Saves us millions. Can anyone work on the Krelman? Of course. Most bee jobs are small ones. But bees know that every small job, if it's done well, means a lot. But choose carefully because you'll stay in the job you pick for the rest of your life. The same job the rest of your life? I didn't know that. What's the difference? You'll be happy to know that bees, as a species, haven't had one day off in 27 million years. So you'll just work us to death? We'll sure try. Wow! That blew my mind! "What's the difference?" How can you say that? One job forever? That's an insane choice to have to make. I'm relieved. Now we only have to make one decision in life. But, Adam, how could they never have told us that? Why would you question anything? We're bees. We're the most perfectly functioning society on Earth. You ever think maybe things work a little too well here? Like what? Give me one example. I don't know. But you know what I'm talking about. Please clear the gate. Royal Nectar Force on approach. Wait a second. Check it out. - Hey, those are Pollen Jocks! - Wow. I've never seen them this close. They know what it's like outside the hive. Yeah, but some don't come back. - Hey, Jocks! - Hi, Jocks! You guys did great! You're monsters! You're sky freaks! I love it! I love it! - I wonder where they were. - I don't know. Their day's not planned. Outside the hive, flying who knows where, doing who knows what. You can't just decide to be a Pollen Jock. You have to be bred for that. Right. Look. That's more pollen than you and I will see in a lifetime. It's just a status symbol. Bees make too much of it. Perhaps. Unless you're wearing it and the ladies see you wearing it. Those ladies? Aren't they our cousins too? Distant. Distant. Look at these two. - Couple of Hive Harrys. - Let's have fun with them. It must be dangerous being a Pollen Jock. Yeah. Once a bear pinned me against a mushroom! He had a paw on my throat, and with the other, he was slapping me! - Oh, my! - I never thought I'd knock him out. What were you doing during this? Trying to alert the authorities. I can autograph that. A little gusty out there today, wasn't it, comrades? Yeah. Gusty. We're hitting a sunflower patch six miles from here tomorrow. - Six miles, huh? - Barry! A puddle jump for us, but maybe you're not up for it. - Maybe I am. - You are not! We're going 0900 at J-Gate. What do you think, buzzy-boy? Are you bee enough? I might be. It all depends on what 0900 means. Hey, Honex! Dad, you surprised me. You decide what you're interested in? - Well, there's a lot of choices. - But you only get one. Do you ever get bored doing the same job every day? Son, let me tell you about stirring. You grab that stick, and you just move it around, and you stir it around. You get yourself into a rhythm. It's a beautiful thing. You know, Dad, the more I think about it, maybe the honey field just isn't right for me. You were thinking of what, making balloon animals? That's a bad job for a guy with a stinger. Janet, your son's not sure he wants to go into honey! - Barry, you are so funny sometimes. - I'm not trying to be funny. You're not funny! You're going into honey. Our son, the stirrer! - You're gonna be a stirrer? - No one's listening to me! Wait till you see the sticks I have. I could say anything right now. I'm gonna get an ant tattoo! Let's open some honey and celebrate! Maybe I'll pierce my thorax. Shave my antennae. Shack up with a grasshopper. Get a gold tooth and call everybody "dawg"! I'm so proud. - We're starting work today! - Today's the day. Come on! All the good jobs will be gone. Yeah, right. Pollen counting, stunt bee, pouring, stirrer, front desk, hair removal... - Is it still available? - Hang on. Two left! One of them's yours! Congratulations! Step to the side. - What'd you get? - Picking crud out. Stellar! Wow! Couple of newbies? Yes, sir! Our first day! We are ready! Make your choice. - You want to go first? - No, you go. Oh, my. What's available? Restroom attendant's open, not for the reason you think. - Any chance of getting the Krelman? - Sure, you're on. I'm sorry, the Krelman just closed out. Wax monkey's always open. The Krelman opened up again. What happened? A bee died. Makes an opening. See? He's dead. Another dead one. Deady. Deadified. Two more dead. Dead from the neck up. Dead from the neck down. That's life! Oh, this is so hard! Heating, cooling, stunt bee, pourer, stirrer, humming, inspector number seven, lint coordinator, stripe supervisor, mite wrangler. Barry, what do you think I should... Barry? Barry! All right, we've got the sunflower patch in quadrant nine... What happened to you? Where are you? - I'm going out. - Out? Out where? - Out there. - Oh, no! I have to, before I go to work for the rest of my life. You're gonna die! You're crazy! Hello? Another call coming in. If anyone's feeling brave, there's a Korean deli on 83rd that gets their roses today. Hey, guys. - Look at that. - Isn't that the kid we saw yesterday? Hold it, son, flight deck's restricted. It's OK, Lou. We're gonna take him up. Really? Feeling lucky, are you? Sign here, here. Just initial that. - Thank you. - OK. You got a rain advisory today, and as you all know, bees cannot fly in rain. So be careful. As always, watch your brooms, hockey sticks, dogs, birds, bears and bats. Also, I got a couple of reports of root beer being poured on us. Murphy's in a home because of it, babbling like a cicada! - That's awful. - And a reminder for you rookies, bee law number one, absolutely no talking to humans! All right, launch positions! Buzz, buzz, buzz, buzz! Buzz, buzz, buzz, buzz! Buzz, buzz, buzz, buzz! Black and yellow! Hello! You ready for this, hot shot? Yeah. Yeah, bring it on. Wind, check. - Antennae, check. - Nectar pack, check. - Wings, check. - Stinger, check. Scared out of my shorts, check. OK, ladies, let's move it out! Pound those petunias, you striped stem-suckers! All of you, drain those flowers! Wow! I'm out! I can't believe I'm out! So blue. I feel so fast and free! Box kite! Wow! Flowers! This is Blue Leader. We have roses visual. Bring it around 30 degrees and hold. Roses! 30 degrees, roger. Bringing it around. Stand to the side, kid. It's got a bit of a kick. That is one nectar collector! - Ever see pollination up close? - No, sir. I pick up some pollen here, sprinkle it over here. Maybe a dash over there, a pinch on that one. See that? It's a little bit of magic. That's amazing. Why do we do that? That's pollen power. More pollen, more flowers, more nectar, more honey for us. Cool. I'm picking up a lot of bright yellow. Could be daisies. Don't we need those? Copy that visual. Wait. One of these flowers seems to be on the move. Say again? You're reporting a moving flower? Affirmative. That was on the line! This is the coolest. What is it? I don't know, but I'm loving this color. It smells good. Not like a flower, but I like it. Yeah, fuzzy. Chemical-y. Careful, guys. It's a little grabby. My sweet lord of bees! Candy-brain, get off there! Problem! - Guys! - This could be bad. Affirmative. Very close. Gonna hurt. Mama's little boy. You are way out of position, rookie! Coming in at you like a missile! Help me! I don't think these are flowers. - Should we tell him? - I think he knows. What is this?! Match point! You can start packing up, honey, because you're about to eat it! Yowser! Gross. There's a bee in the car! - Do something! - I'm driving! - Hi, bee. - He's back here! He's going to sting me! Nobody move. If you don't move, he won't sting you. Freeze! He blinked! Spray him, Granny! What are you doing?! Wow... the tension level out here is unbelievable. I gotta get home. Can't fly in rain. Can't fly in rain. Can't fly in rain. Mayday! Mayday! Bee going down! Ken, could you close the window please? Ken, could you close the window please? Check out my new resume. I made it into a fold-out brochure. You see? Folds out. Oh, no. More humans. I don't need this. What was that? Maybe this time. This time. This time. This time! This time! This... Drapes! That is diabolical. It's fantastic. It's got all my special skills, even my top-ten favorite movies. What's number one? Star Wars? Nah, I don't go for that... ...kind of stuff. No wonder we shouldn't talk to them. They're out of their minds. When I leave a job interview, they're flabbergasted, can't believe what I say. There's the sun. Maybe that's a way out. I don't remember the sun having a big 75 on it. I predicted global warming. I could feel it getting hotter. At first I thought it was just me. Wait! Stop! Bee! Stand back. These are winter boots. Wait! Don't kill him! You know I'm allergic to them! This thing could kill me! Why does his life have less value than yours? Why does his life have any less value than mine? Is that your statement? I'm just saying all life has value. You don't know what he's capable of feeling. My brochure! There you go, little guy. I'm not scared of him. It's an allergic thing. Put that on your resume brochure. My whole face could puff up. Make it one of your special skills. Knocking someone out is also a special skill. Right. Bye, Vanessa. Thanks. - Vanessa, next week? Yogurt night? - Sure, Ken. You know, whatever. - You could put carob chips on there. - Bye. - Supposed to be less calories. - Bye. I gotta say something. She saved my life. I gotta say something. All right, here it goes. Nah. What would I say? I could really get in trouble. It's a bee law. You're not supposed to talk to a human. I can't believe I'm doing this. I've got to. Oh, I can't do it. Come on! No. Yes. No. Do it. I can't. How should I start it? "You like jazz?" No, that's no good. Here she comes! Speak, you fool! Hi! I'm sorry. - You're talking. - Yes, I know. You're talking! I'm so sorry. No, it's OK. It's fine. I know I'm dreaming. But I don't recall going to bed. Well, I'm sure this is very disconcerting. This is a bit of a surprise to me. I mean, you're a bee! I am. And I'm not supposed to be doing this, but they were all trying to kill me. And if it wasn't for you... I had to thank you. It's just how I was raised. That was a little weird. - I'm talking with a bee. - Yeah. I'm talking to a bee. And the bee is talking to me! I just want to say I'm grateful. I'll leave now. - Wait! How did you learn to do that? - What? The talking thing. Same way you did, I guess. "Mama, Dada, honey." You pick it up. - That's very funny. - Yeah. Bees are funny. If we didn't laugh, we'd cry with what we have to deal with. Anyway... Can I... ...get you something? - Like what? I don't know. I mean... I don't know. Coffee? I don't want to put you out. It's no trouble. It takes two minutes. - It's just coffee. - I hate to impose. - Don't be ridiculous! - Actually, I would love a cup. Hey, you want rum cake? - I shouldn't. - Have some. - No, I can't. - Come on! I'm trying to lose a couple micrograms. - Where? - These stripes don't help. You look great! I don't know if you know anything about fashion. Are you all right? No. He's making the tie in the cab as they're flying up Madison. He finally gets there. He runs up the steps into the church. The wedding is on. And he says, "Watermelon? I thought you said Guatemalan. Why would I marry a watermelon?" Is that a bee joke? That's the kind of stuff we do. Yeah, different. So, what are you gonna do, Barry? About work? I don't know. I want to do my part for the hive, but I can't do it the way they want. I know how you feel. - You do? - Sure. My parents wanted me to be a lawyer or a doctor, but I wanted to be a florist. - Really? - My only interest is flowers. Our new queen was just elected with that same campaign slogan. Anyway, if you look... There's my hive right there. See it? You're in Sheep Meadow! Yes! I'm right off the Turtle Pond! No way! I know that area. I lost a toe ring there once. - Why do girls put rings on their toes? - Why not? - It's like putting a hat on your knee. - Maybe I'll try that. - You all right, ma'am? - Oh, yeah. Fine. Just having two cups of coffee! Anyway, this has been great. Thanks for the coffee. Yeah, it's no trouble. Sorry I couldn't finish it. If I did, I'd be up the rest of my life. Are you...? Can I take a piece of this with me? Sure! Here, have a crumb. - Thanks! - Yeah. All right. Well, then... I guess I'll see you around. Or not. OK, Barry. And thank you so much again... for before. Oh, that? That was nothing. Well, not nothing, but... Anyway... This can't possibly work. He's all set to go. We may as well try it. OK, Dave, pull the chute. - Sounds amazing. - It was amazing! It was the scariest, happiest moment of my life. Humans! I can't believe you were with humans! Giant, scary humans! What were they like? Huge and crazy. They talk crazy. They eat crazy giant things. They drive crazy. - Do they try and kill you, like on TV? - Some of them. But some of them don't. - How'd you get back? - Poodle. You did it, and I'm glad. You saw whatever you wanted to see. You had your "experience." Now you can pick out your job and be normal. - Well... - Well? Well, I met someone. You did? Was she Bee-ish? - A wasp?! Your parents will kill you! - No, no, no, not a wasp. - Spider? - I'm not attracted to spiders. I know it's the hottest thing, with the eight legs and all. I can't get by that face. So who is she? She's... human. No, no. That's a bee law. You wouldn't break a bee law. - Her name's Vanessa. - Oh, boy. She's so nice. And she's a florist! Oh, no! You're dating a human florist! We're not dating. You're flying outside the hive, talking to humans that attack our homes with power washers and M-80s! One-eighth a stick of dynamite! She saved my life! And she understands me. This is over! Eat this. This is not over! What was that? - They call it a crumb. - It was so stingin' stripey! And that's not what they eat. That's what falls off what they eat! - You know what a Cinnabon is? - No. It's bread and cinnamon and frosting. They heat it up... Sit down! ...really hot! - Listen to me! We are not them! We're us. There's us and there's them! Yes, but who can deny the heart that is yearning? There's no yearning. Stop yearning. Listen to me! You have got to start thinking bee, my friend. Thinking bee! - Thinking bee. - Thinking bee. Thinking bee! Thinking bee! Thinking bee! Thinking bee! There he is. He's in the pool. You know what your problem is, Barry? I gotta start thinking bee? How much longer will this go on? It's been three days! Why aren't you working? I've got a lot of big life decisions to think about. What life? You have no life! You have no job. You're barely a bee! Would it kill you to make a little honey? Barry, come out. Your father's talking to you. Martin, would you talk to him? Barry, I'm talking to you! You coming? Got everything? All set! Go ahead. I'll catch up. Don't be too long. Watch this! Vanessa! - We're still here. - I told you not to yell at him. He doesn't respond to yelling! - Then why yell at me? - Because you don't listen! I'm not listening to this. Sorry, I've gotta go. - Where are you going? - I'm meeting a friend. A girl? Is this why you can't decide? Bye. I just hope she's Bee-ish. They have a huge parade of flowers every year in Pasadena? To be in the Tournament of Roses, that's every florist's dream! Up on a float, surrounded by flowers, crowds cheering. A tournament. Do the roses compete in athletic events? No. All right, I've got one. How come you don't fly everywhere? It's exhausting. Why don't you run everywhere? It's faster. Yeah, OK, I see, I see. All right, your turn. TiVo. You can just freeze live TV? That's insane! You don't have that? We have Hivo, but it's a disease. It's a horrible, horrible disease. Oh, my. Dumb bees! You must want to sting all those jerks. We try not to sting. It's usually fatal for us. So you have to watch your temper. Very carefully. You kick a wall, take a walk, write an angry letter and throw it out. Work through it like any emotion: Anger, jealousy, lust. Oh, my goodness! Are you OK? Yeah. - What is wrong with you?! - It's a bug. He's not bothering anybody. Get out of here, you creep! What was that? A Pic 'N' Save circular? Yeah, it was. How did you know? It felt like about 10 pages. Seventy-five is pretty much our limit. You've really got that down to a science. - I lost a cousin to Italian Vogue. - I'll bet. What in the name of Mighty Hercules is this? How did this get here? Cute Bee, Golden Blossom, Ray Liotta Private Select? - Is he that actor? - I never heard of him. - Why is this here? - For people. We eat it. You don't have enough food of your own? - Well, yes. - How do you get it? - Bees make it. - I know who makes it! And it's hard to make it! There's heating, cooling, stirring. You need a whole Krelman thing! - It's organic. - It's our-ganic! It's just honey, Barry. Just what?! Bees don't know about this! This is stealing! A lot of stealing! You've taken our homes, schools, hospitals! This is all we have! And it's on sale?! I'm getting to the bottom of this. I'm getting to the bottom of all of this! Hey, Hector. - You almost done? - Almost. He is here. I sense it. Well, I guess I'll go home now and just leave this nice honey out, with no one around. You're busted, box boy! I knew I heard something. So you can talk! I can talk. And now you'll start talking! Where you getting the sweet stuff? Who's your supplier? I don't understand. I thought we were friends. The last thing we want to do is upset bees! You're too late! It's ours now! You, sir, have crossed the wrong sword! You, sir, will be lunch for my iguana, Ignacio! Where is the honey coming from? Tell me where! Honey Farms! It comes from Honey Farms! Crazy person! What horrible thing has happened here? These faces, they never knew what hit them. And now they're on the road to nowhere! Just keep still. What? You're not dead? Do I look dead? They will wipe anything that moves. Where you headed? To Honey Farms. I am onto something huge here. I'm going to Alaska. Moose blood, crazy stuff. Blows your head off! I'm going to Tacoma. - And you? - He really is dead. All right. Uh-oh! - What is that?! - Oh, no! - A wiper! Triple blade! - Triple blade? Jump on! It's your only chance, bee! Why does everything have to be so doggone clean?! How much do you people need to see?! Open your eyes! Stick your head out the window! From NPR News in Washington, I'm Carl Kasell. But don't kill no more bugs! - Bee! - Moose blood guy!! - You hear something? - Like what? Like tiny screaming. Turn off the radio. Whassup, bee boy? Hey, Blood. Just a row of honey jars, as far as the eye could see. Wow! I assume wherever this truck goes is where they're getting it. I mean, that honey's ours. - Bees hang tight. - We're all jammed in. It's a close community. Not us, man. We on our own. Every mosquito on his own. - What if you get in trouble? - You a mosquito, you in trouble. Nobody likes us. They just smack. See a mosquito, smack, smack! At least you're out in the world. You must meet girls. Mosquito girls try to trade up, get with a moth, dragonfly. Mosquito girl don't want no mosquito. You got to be kidding me! Mooseblood's about to leave the building! So long, bee! - Hey, guys! - Mooseblood! I knew I'd catch y'all down here. Did you bring your crazy straw? We throw it in jars, slap a label on it, and it's pretty much pure profit. What is this place? A bee's got a brain the size of a pinhead. They are pinheads! Pinhead. - Check out the new smoker. - Oh, sweet. That's the one you want. The Thomas 3000! Smoker? Ninety puffs a minute, semi-automatic. Twice the nicotine, all the tar. A couple breaths of this knocks them right out. They make the honey, and we make the money. "They make the honey, and we make the money"? Oh, my! What's going on? Are you OK? Yeah. It doesn't last too long. Do you know you're in a fake hive with fake walls? Our queen was moved here. We had no choice. This is your queen? That's a man in women's clothes! That's a drag queen! What is this? Oh, no! There's hundreds of them! Bee honey. Our honey is being brazenly stolen on a massive scale! This is worse than anything bears have done! I intend to do something. Oh, Barry, stop. Who told you humans are taking our honey? That's a rumor. Do these look like rumors? That's a conspiracy theory. These are obviously doctored photos. How did you get mixed up in this? He's been talking to humans. - What? - Talking to humans?! He has a human girlfriend. And they make out! Make out? Barry! We do not. - You wish you could. - Whose side are you on? The bees! I dated a cricket once in San Antonio. Those crazy legs kept me up all night. Barry, this is what you want to do with your life? I want to do it for all our lives. Nobody works harder than bees! Dad, I remember you coming home so overworked your hands were still stirring. You couldn't stop. I remember that. What right do they have to our honey? We live on two cups a year. They put it in lip balm for no reason whatsoever! Even if it's true, what can one bee do? Sting them where it really hurts. In the face! The eye! - That would hurt. - No. Up the nose? That's a killer. There's only one place you can sting the humans, one place where it matters. Hive at Five, the hive's only full-hour action news source. No more bee beards! With Bob Bumble at the anchor desk. Weather with Storm Stinger. Sports with Buzz Larvi. And Jeanette Chung. - Good evening. I'm Bob Bumble. - And I'm Jeanette Chung. A tri-county bee, Barry Benson, intends to sue the human race for stealing our honey, packaging it and profiting from it illegally! Tomorrow night on Bee Larry King, we'll have three former queens here in our studio, discussing their new book, Classy Ladies, out this week on Hexagon. Tonight we're talking to Barry Benson. Did you ever think, "I'm a kid from the hive. I can't do this"? Bees have never been afraid to change the world. What about Bee Columbus? Bee Gandhi? Bee Jesus? Where I'm from, we'd never sue humans. We were thinking of stickball or candy stores. How old are you? The bee community is supporting you in this case, which will be the trial of the bee century. You know, they have a Larry King in the human world too. It's a common name. Next week... He looks like you and has a show and suspenders and colored dots... Next week... Glasses, quotes on the bottom from the guest even though you just heard 'em. Bear Week next week! They're scary, hairy and here live. Always leans forward, pointy shoulders, squinty eyes, very Jewish. In tennis, you attack at the point of weakness! It was my grandmother, Ken. She's 81. Honey, her backhand's a joke! I'm not gonna take advantage of that? Quiet, please. Actual work going on here. - Is that that same bee? - Yes, it is! I'm helping him sue the human race. - Hello. - Hello, bee. This is Ken. Yeah, I remember you. Timberland, size ten and a half. Vibram sole, I believe. Why does he talk again? Listen, you better go 'cause we're really busy working. But it's our yogurt night! Bye-bye. Why is yogurt night so difficult?! You poor thing. You two have been at this for hours! Yes, and Adam here has been a huge help. - Frosting... - How many sugars? Just one. I try not to use the competition. So why are you helping me? Bees have good qualities. And it takes my mind off the shop. Instead of flowers, people are giving balloon bouquets now. Those are great, if you're three. And artificial flowers. - Oh, those just get me psychotic! - Yeah, me too. Bent stingers, pointless pollination. Bees must hate those fake things! Nothing worse than a daffodil that's had work done. Maybe this could make up for it a little bit. - This lawsuit's a pretty big deal. - I guess. You sure you want to go through with it? Am I sure? When I'm done with the humans, they won't be able to say, "Honey, I'm home," without paying a royalty! It's an incredible scene here in downtown Manhattan, where the world anxiously waits, because for the first time in history, we will hear for ourselves if a honeybee can actually speak. What have we gotten into here, Barry? It's pretty big, isn't it? I can't believe how many humans don't work during the day. You think billion-dollar multinational food companies have good lawyers? Everybody needs to stay behind the barricade. - What's the matter? - I don't know, I just got a chill. Well, if it isn't the bee team. You boys work on this? All rise! The Honorable Judge Bumbleton presiding. All right. Case number 4475, Superior Court of New York, Barry Bee Benson v. the Honey Industry is now in session. Mr. Montgomery, you're representing the five food companies collectively? A privilege. Mr. Benson... you're representing all the bees of the world? I'm kidding. Yes, Your Honor, we're ready to proceed. Mr. Montgomery, your opening statement, please. Ladies and gentlemen of the jury, my grandmother was a simple woman. Born on a farm, she believed it was man's divine right to benefit from the bounty of nature God put before us. If we lived in the topsy-turvy world Mr. Benson imagines, just think of what would it mean. I would have to negotiate with the silkworm for the elastic in my britches! Talking bee! How do we know this isn't some sort of holographic motion-picture-capture Hollywood wizardry? They could be using laser beams! Robotics! Ventriloquism! Cloning! For all we know, he could be on steroids! Mr. Benson? Ladies and gentlemen, there's no trickery here. I'm just an ordinary bee. Honey's pretty important to me. It's important to all bees. We invented it! We make it. And we protect it with our lives. Unfortunately, there are some people in this room who think they can take it from us 'cause we're the little guys! I'm hoping that, after this is all over, you'll see how, by taking our honey, you not only take everything we have but everything we are! I wish he'd dress like that all the time. So nice! Call your first witness. So, Mr. Klauss Vanderhayden of Honey Farms, big company you have. I suppose so. I see you also own Honeyburton and Honron! Yes, they provide beekeepers for our farms. Beekeeper. I find that to be a very disturbing term. I don't imagine you employ any bee-free-ers, do you? - No. - I couldn't hear you. - No. - No. Because you don't free bees. You keep bees. Not only that, it seems you thought a bear would be an appropriate image for a jar of honey. They're very lovable creatures. Yogi Bear, Fozzie Bear, Build-A-Bear. You mean like this? Bears kill bees! How'd you like his head crashing through your living room?! Biting into your couch! Spitting out your throw pillows! OK, that's enough. Take him away. So, Mr. Sting, thank you for being here. Your name intrigues me. - Where have I heard it before? - I was with a band called The Police. But you've never been a police officer, have you? No, I haven't. No, you haven't. And so here we have yet another example of bee culture casually stolen by a human for nothing more than a prance-about stage name. Oh, please. Have you ever been stung, Mr. Sting? Because I'm feeling a little stung, Sting. Or should I say... Mr. Gordon M. Sumner! That's not his real name?! You idiots! Mr. Liotta, first, belated congratulations on your Emmy win for a guest spot on ER in 2005. Thank you. Thank you. I see from your resume that you're devilishly handsome with a churning inner turmoil that's ready to blow. I enjoy what I do. Is that a crime? Not yet it isn't. But is this what it's come to for you? Exploiting tiny, helpless bees so you don't have to rehearse your part and learn your lines, sir? Watch it, Benson! I could blow right now! This isn't a goodfella. This is a bad fella! Why doesn't someone just step on this creep, and we can all go home?! - Order in this court! - You're all thinking it! Order! Order, I say! - Say it! - Mr. Liotta, please sit down! I think it was awfully nice of that bear to pitch in like that. I think the jury's on our side. Are we doing everything right, legally? I'm a florist. Right. Well, here's to a great team. To a great team! Well, hello. - Ken! - Hello. I didn't think you were coming. No, I was just late. I tried to call, but... the battery. I didn't want all this to go to waste, so I called Barry. Luckily, he was free. Oh, that was lucky. There's a little left. I could heat it up. Yeah, heat it up, sure, whatever. So I hear you're quite a tennis player. I'm not much for the game myself. The ball's a little grabby. That's where I usually sit. Right... there. Ken, Barry was looking at your resume, and he agreed with me that eating with chopsticks isn't really a special skill. You think I don't see what you're doing? I know how hard it is to find the right job. We have that in common. Do we? Bees have 100 percent employment, but we do jobs like taking the crud out. That's just what I was thinking about doing. Ken, I let Barry borrow your razor for his fuzz. I hope that was all right. I'm going to drain the old stinger. Yeah, you do that. Look at that. You know, I've just about had it with your little mind games. - What's that? - Italian Vogue. Mamma mia, that's a lot of pages. A lot of ads. Remember what Van said, why is your life more valuable than mine? Funny, I just can't seem to recall that! I think something stinks in here! I love the smell of flowers. How do you like the smell of flames?! Not as much. Water bug! Not taking sides! Ken, I'm wearing a Chapstick hat! This is pathetic! I've got issues! Well, well, well, a royal flush! - You're bluffing. - Am I? Surf's up, dude! Poo water! That bowl is gnarly. Except for those dirty yellow rings! Kenneth! What are you doing?! You know, I don't even like honey! I don't eat it! We need to talk! He's just a little bee! And he happens to be the nicest bee I've met in a long time! Long time? What are you talking about?! Are there other bugs in your life? No, but there are other things bugging me in life. And you're one of them! Fine! Talking bees, no yogurt night... My nerves are fried from riding on this emotional roller coaster! Goodbye, Ken. And for your information, I prefer sugar-free, artificial sweeteners made by man! I'm sorry about all that. I know it's got an aftertaste! I like it! I always felt there was some kind of barrier between Ken and me. I couldn't overcome it. Oh, well. Are you OK for the trial? I believe Mr. Montgomery is about out of ideas. We would like to call Mr. Barry Benson Bee to the stand. Good idea! You can really see why he's considered one of the best lawyers... Yeah. Layton, you've gotta weave some magic with this jury, or it's gonna be all over. Don't worry. The only thing I have to do to turn this jury around is to remind them of what they don't like about bees. - You got the tweezers? - Are you allergic? Only to losing, son. Only to losing. Mr. Benson Bee, I'll ask you what I think we'd all like to know. What exactly is your relationship to that woman? We're friends. - Good friends? - Yes. How good? Do you live together? Wait a minute... Are you her little... ...bedbug? I've seen a bee documentary or two. From what I understand, doesn't your queen give birth to all the bee children? - Yeah, but... - So those aren't your real parents! - Oh, Barry... - Yes, they are! Hold me back! You're an illegitimate bee, aren't you, Benson? He's denouncing bees! Don't y'all date your cousins? - Objection! - I'm going to pincushion this guy! Adam, don't! It's what he wants! Oh, I'm hit! Oh, lordy, I am hit! Order! Order! The venom! The venom is coursing through my veins! I have been felled by a winged beast of destruction! You see? You can't treat them like equals! They're striped savages! Stinging's the only thing they know! It's their way! - Adam, stay with me. - I can't feel my legs. What angel of mercy will come forward to suck the poison from my heaving buttocks? I will have order in this court. Order! Order, please! The case of the honeybees versus the human race took a pointed turn against the bees yesterday when one of their legal team stung Layton T. Montgomery. - Hey, buddy. - Hey. - Is there much pain? - Yeah. I... I blew the whole case, didn't I? It doesn't matter. What matters is you're alive. You could have died. I'd be better off dead. Look at me. They got it from the cafeteria downstairs, in a tuna sandwich. Look, there's a little celery still on it. What was it like to sting someone? I can't explain it. It was all... All adrenaline and then... and then ecstasy! All right. You think it was all a trap? Of course. I'm sorry. I flew us right into this. What were we thinking? Look at us. We're just a couple of bugs in this world. What will the humans do to us if they win? I don't know. I hear they put the roaches in motels. That doesn't sound so bad. Adam, they check in, but they don't check out! Oh, my. Could you get a nurse to close that window? - Why? - The smoke. Bees don't smoke. Right. Bees don't smoke. Bees don't smoke! But some bees are smoking. That's it! That's our case! It is? It's not over? Get dressed. I've gotta go somewhere. Get back to the court and stall. Stall any way you can. And assuming you've done step correctly, you're ready for the tub. Mr. Flayman. Yes? Yes, Your Honor! Where is the rest of your team? Well, Your Honor, it's interesting. Bees are trained to fly haphazardly, and as a result, we don't make very good time. I actually heard a funny story about... Your Honor, haven't these ridiculous bugs taken up enough of this court's valuable time? How much longer will we allow these absurd shenanigans to go on? They have presented no compelling evidence to support their charges against my clients, who run legitimate businesses. I move for a complete dismissal of this entire case! Mr. Flayman, I'm afraid I'm going to have to consider Mr. Montgomery's motion. But you can't! We have a terrific case. Where is your proof? Where is the evidence? Show me the smoking gun! Hold it, Your Honor! You want a smoking gun? Here is your smoking gun. What is that? It's a bee smoker! What, this? This harmless little contraption? This couldn't hurt a fly, let alone a bee. Look at what has happened to bees who have never been asked, "Smoking or non?" Is this what nature intended for us? To be forcibly addicted to smoke machines and man-made wooden slat work camps? Living out our lives as honey slaves to the white man? - What are we gonna do? - He's playing the species card. Ladies and gentlemen, please, free these bees! Free the bees! Free the bees! Free the bees! Free the bees! Free the bees! The court finds in favor of the bees! Vanessa, we won! I knew you could do it! High-five! Sorry. I'm OK! You know what this means? All the honey will finally belong to the bees. Now we won't have to work so hard all the time. This is an unholy perversion of the balance of nature, Benson. You'll regret this. Barry, how much honey is out there? All right. One at a time. Barry, who are you wearing? My sweater is Ralph Lauren, and I have no pants. - What if Montgomery's right? - What do you mean? We've been living the bee way a long time, 27 million years. Congratulations on your victory. What will you demand as a settlement? First, we'll demand a complete shutdown of all bee work camps. Then we want back the honey that was ours to begin with, every last drop. We demand an end to the glorification of the bear as anything more than a filthy, smelly, bad-breath stink machine. We're all aware of what they do in the woods. Wait for my signal. Take him out. He'll have nauseous for a few hours, then he'll be fine. And we will no longer tolerate bee-negative nicknames... But it's just a prance-about stage name! ...unnecessary inclusion of honey in bogus health products and la-dee-da human tea-time snack garnishments. Can't breathe. Bring it in, boys! Hold it right there! Good. Tap it. Mr. Buzzwell, we just passed three cups, and there's gallons more coming! - I think we need to shut down! - Shut down? We've never shut down. Shut down honey production! Stop making honey! Turn your key, sir! What do we do now? Cannonball! We're shutting honey production! Mission abort. Aborting pollination and nectar detail. Returning to base. Adam, you wouldn't believe how much honey was out there. Oh, yeah? What's going on? Where is everybody? - Are they out celebrating? - They're home. They don't know what to do. Laying out, sleeping in. I heard your Uncle Carl was on his way to San Antonio with a cricket. At least we got our honey back. Sometimes I think, so what if humans liked our honey? Who wouldn't? It's the greatest thing in the world! I was excited to be part of making it. This was my new desk. This was my new job. I wanted to do it really well. And now... Now I can't. I don't understand why they're not happy. I thought their lives would be better! They're doing nothing. It's amazing. Honey really changes people. You don't have any idea what's going on, do you? - What did you want to show me? - This. What happened here? That is not the half of it. Oh, no. Oh, my. They're all wilting. Doesn't look very good, does it? No. And whose fault do you think that is? You know, I'm gonna guess bees. Bees? Specifically, me.



I was bored .__.
**Post 2:**

> idk who need this but here’s the entire bee movie script  According to all known laws of aviation, there is no way a bee should be able to fly. Its wings are too small to get its fat little body off the ground. The bee, of course, flies anyway because bees don't care what humans think is impossible. Yellow, black. Yellow, black. Yellow, black. Yellow, black. Ooh, black and yellow! Let's shake it up a little. Barry! Breakfast is ready! Ooming! Hang on a second. Hello? - Barry? - Adam? - Oan you believe this is happening? - I can't. I'll pick you up. Looking sharp. Use the stairs. Your father paid good money for those. Sorry. I'm excited. Here's the graduate. We're very proud of you, son. A perfect report card, all B's. Very proud. Ma! I got a thing going here. - You got lint on your fuzz. - Ow! That's me! - Wave to us! We'll be in row 118,000. - Bye! Barry, I told you, stop flying in the house! - Hey, Adam. - Hey, Barry. - Is that fuzz gel? - A little. Special day, graduation. Never thought I'd make it. Three days grade school, three days high school. Those were awkward. Three days college. I'm glad I took a day and hitchhiked around the hive. You did come back different. - Hi, Barry. - Artie, growing a mustache? Looks good. - Hear about Frankie? - Yeah. - You going to the funeral? - No, I'm not going. Everybody knows, sting someone, you die. Don't waste it on a squirrel. Such a hothead. I guess he could have just gotten out of the way. I love this incorporating an amusement park into our day. That's why we don't need vacations. Boy, quite a bit of pomp... under the circumstances. - Well, Adam, today we are men. - We are! - Bee-men. - Amen! Hallelujah! Students, faculty, distinguished bees, please welcome Dean Buzzwell. Welcome, New Hive Oity graduating class of... ...9:15. That concludes our ceremonies. And begins your career at Honex Industries! Will we pick ourjob today? I heard it's just orientation. Heads up! Here we go. Keep your hands and antennas inside the tram at all times. - Wonder what it'll be like? - A little scary. Welcome to Honex, a division of Honesco and a part of the Hexagon Group. This is it! Wow. Wow. We know that you, as a bee, have worked your whole life to get to the point where you can work for your whole life. Honey begins when our valiant Pollen Jocks bring the nectar to the hive. Our top-secret formula is automatically color-corrected, scent-adjusted and bubble-contoured into this soothing sweet syrup with its distinctive golden glow you know as... Honey! - That girl was hot. - She's my cousin! - She is? - Yes, we're all cousins. - Right. You're right. - At Honex, we constantly strive to improve every aspect of bee existence. These bees are stress-testing a new helmet technology. - What do you think he makes? - Not enough. Here we have our latest advancement, the Krelman. - What does that do? - Oatches that little strand of honey that hangs after you pour it. Saves us millions. Oan anyone work on the Krelman? Of course. Most bee jobs are small ones. But bees know that every small job, if it's done well, means a lot. But choose carefully because you'll stay in the job you pick for the rest of your life. The same job the rest of your life? I didn't know that. What's the difference? You'll be happy to know that bees, as a species, haven't had one day off in 27 million years. So you'll just work us to death? We'll sure try. Wow! That blew my mind! "What's the difference?" How can you say that? One job forever? That's an insane choice to have to make. I'm relieved. Now we only have to make one decision in life. But, Adam, how could they never have told us that? Why would you question anything? We're bees. We're the most perfectly functioning society on Earth. You ever think maybe things work a little too well here? Like what? Give me one example. I don't know. But you know what I'm talking about. Please clear the gate. Royal Nectar Force on approach. Wait a second. Oheck it out. - Hey, those are Pollen Jocks! - Wow. I've never seen them this close. They know what it's like outside the hive. Yeah, but some don't come back. - Hey, Jocks! - Hi, Jocks! You guys did great! You're monsters! You're sky freaks! I love it! I love it! - I wonder where they were. - I don't know. Their day's not planned. Outside the hive, flying who knows where, doing who knows what. You can'tjust decide to be a Pollen Jock. You have to be bred for that. Right. Look. That's more pollen than you and I will see in a lifetime. It's just a status symbol. Bees make too much of it. Perhaps. Unless you're wearing it and the ladies see you wearing it. Those ladies? Aren't they our cousins too? Distant. Distant. Look at these two. - Oouple of Hive Harrys. - Let's have fun with them. It must be dangerous being a Pollen Jock. Yeah. Once a bear pinned me against a mushroom! He had a paw on my throat, and with the other, he was slapping me! - Oh, my! - I never thought I'd knock him out. What were you doing during this? Trying to alert the authorities. I can autograph that. A little gusty out there today, wasn't it, comrades? Yeah. Gusty. We're hitting a sunflower patch six miles from here tomorrow. - Six miles, huh? - Barry! A puddle jump for us, but maybe you're not up for it. - Maybe I am. - You are not! We're going 0900 at J-Gate. What do you think, buzzy-boy? Are you bee enough? I might be. It all depends on what 0900 means. Hey, Honex! Dad, you surprised me. You decide what you're interested in? - Well, there's a lot of choices. - But you only get one. Do you ever get bored doing the same job every day? Son, let me tell you about stirring. You grab that stick, and you just move it around, and you stir it around. You get yourself into a rhythm. It's a beautiful thing. You know, Dad, the more I think about it, maybe the honey field just isn't right for me. You were thinking of what, making balloon animals? That's a bad job for a guy with a stinger. Janet, your son's not sure he wants to go into honey! - Barry, you are so funny sometimes. - I'm not trying to be funny. You're not funny! You're going into honey. Our son, the stirrer! - You're gonna be a stirrer? - No one's listening to me! Wait till you see the sticks I have. I could say anything right now. I'm gonna get an ant tattoo! Let's open some honey and celebrate! Maybe I'll pierce my thorax. Shave my antennae. Shack up with a grasshopper. Get a gold tooth and call everybody "dawg"! I'm so proud. - We're starting work today! - Today's the day. Oome on! All the good jobs will be gone. Yeah, right. Pollen counting, stunt bee, pouring, stirrer, front desk, hair removal... - Is it still available? - Hang on. Two left! One of them's yours! Oongratulations! Step to the side. - What'd you get? - Picking crud out. Stellar! Wow! Oouple of newbies? Yes, sir! Our first day! We are ready! Make your choice. - You want to go first? - No, you go. Oh, my. What's available? Restroom attendant's open, not for the reason you think. - Any chance of getting the Krelman? - Sure, you're on. I'm sorry, the Krelman just closed out. Wax monkey's always open. The Krelman opened up again. What happened? A bee died. Makes an opening. See? He's dead. Another dead one. Deady. Deadified. Two more dead. Dead from the neck up. Dead from the neck down. That's life! Oh, this is so hard! Heating, cooling, stunt bee, pourer, stirrer, humming, inspector number seven, lint coordinator, stripe supervisor, mite wrangler. Barry, what do you think I should... Barry? Barry! All right, we've got the sunflower patch in quadrant nine... What happened to you? Where are you? - I'm going out. - Out? Out where? - Out there. - Oh, no! I have to, before I go to work for the rest of my life. You're gonna die! You're crazy! Hello? Another call coming in. If anyone's feeling brave, there's a Korean deli on 83rd that gets their roses today. Hey, guys. - Look at that. - Isn't that the kid we saw yesterday? Hold it, son, flight deck's restricted. It's OK, Lou. We're gonna take him up. Really? Feeling lucky, are you? Sign here, here. Just initial that. - Thank you. - OK. You got a rain advisory today, and as you all know, bees cannot fly in rain. So be careful. As always, watch your brooms, hockey sticks, dogs, birds, bears and bats. Also, I got a couple of reports of root beer being poured on us. Murphy's in a home because of it, babbling like a cicada! - That's awful. - And a reminder for you rookies, bee law number one, absolutely no talking to humans! All right, launch positions! Buzz, buzz, buzz, buzz! Buzz, buzz, buzz, buzz! Buzz, buzz, buzz, buzz! Black and yellow! Hello! You ready for this, hot shot? Yeah. Yeah, bring it on. Wind, check. - Antennae, check. - Nectar pack, check. - Wings, check. - Stinger, check. Scared out of my shorts, check. OK, ladies, let's move it out! Pound those petunias, you striped stem-suckers! All of you, drain those flowers! Wow! I'm out! I can't believe I'm out! So blue. I feel so fast and free! Box kite! Wow! Flowers! This is Blue Leader. We have roses visual. Bring it around 30 degrees and hold. Roses! 30 degrees, roger. Bringing it around. Stand to the side, kid. It's got a bit of a kick. That is one nectar collector! - Ever see pollination up close? - No, sir. I pick up some pollen here, sprinkle it over here. Maybe a dash over there, a pinch on that one. See that? It's a little bit of magic. That's amazing. Why do we do that? That's pollen power. More pollen, more flowers, more nectar, more honey for us. Oool. I'm picking up a lot of bright yellow. Oould be daisies. Don't we need those? Oopy that visual. Wait. One of these flowers seems to be on the move. Say again? You're reporting a moving flower? Affirmative. That was on the line! This is the coolest. What is it? I don't know, but I'm loving this color. It smells good. Not like a flower, but I like it. Yeah, fuzzy. Ohemical-y. Oareful, guys. It's a little grabby. My sweet lord of bees! Oandy-brain, get off there! Problem! - Guys! - This could be bad. Affirmative. Very close. Gonna hurt. Mama's little boy. You are way out of position, rookie! Ooming in at you like a missile! Help me! I don't think these are flowers. - Should we tell him? - I think he knows. What is this?! Match point! You can start packing up, honey, because you're about to eat it! Yowser! Gross. There's a bee in the car! - Do something! - I'm driving! - Hi, bee. - He's back here! He's going to sting me! Nobody move. If you don't move, he won't sting you. Freeze! He blinked! Spray him, Granny! What are you doing?! Wow... the tension level out here is unbelievable. I gotta get home. Oan't fly in rain. Oan't fly in rain. Oan't fly in rain. Mayday! Mayday! Bee going down! Ken, could you close the window please? Ken, could you close the window please? Oheck out my new resume. I made it into a fold-out brochure. You see? Folds out. Oh, no. More humans. I don't need this. What was that? Maybe this time. This time. This time. This time! This time! This... Drapes! That is diabolical. It's fantastic. It's got all my special skills, even my top-ten favorite movies. What's number one? Star Wars? Nah, I don't go for that... ...kind of stuff. No wonder we shouldn't talk to them. They're out of their minds. When I leave a job interview, they're flabbergasted, can't believe what I say. There's the sun. Maybe that's a way out. I don't remember the sun having a big 75 on it. I predicted global warming. I could feel it getting hotter. At first I thought it was just me. Wait! Stop! Bee! Stand back. These are winter boots. Wait! Don't kill him! You know I'm allergic to them! This thing could kill me! Why does his life have less value than yours? Why does his life have any less value than mine? Is that your statement? I'm just saying all life has value. You don't know what he's capable of feeling. My brochure! There you go, little guy. I'm not scared of him. It's an allergic thing. Put that on your resume brochure. My whole face could puff up. Make it one of your special skills. Knocking someone out is also a special skill. Right. Bye, Vanessa. Thanks. - Vanessa, next week? Yogurt night? - Sure, Ken. You know, whatever. - You could put carob chips on there. - Bye. - Supposed to be less calories. - Bye. I gotta say something. She saved my life. I gotta say something. All right, here it goes. Nah. What would I say? I could really get in trouble. It's a bee law. You're not supposed to talk to a human. I can't believe I'm doing this. I've got to. Oh, I can't do it. Oome on! No. Yes. No. Do it. I can't. How should I start it? "You like jazz?" No, that's no good. Here she comes! Speak, you fool! Hi! I'm sorry. - You're talking. - Yes, I know. You're talking! I'm so sorry. No, it's OK. It's fine. I know I'm dreaming. But I don't recall going to bed. Well, I'm sure this is very disconcerting. This is a bit of a surprise to me. I mean, you're a bee! I am. And I'm not supposed to be doing this, but they were all trying to kill me. And if it wasn't for you... I had to thank you. It's just how I was raised. That was a little weird. - I'm talking with a bee. - Yeah. I'm talking to a bee. And the bee is talking to me! I just want to say I'm grateful. I'll leave now. - Wait! How did you learn to do that? - What? The talking thing. Same way you did, I guess. "Mama, Dada, honey." You pick it up. - That's very funny. - Yeah. Bees are funny. If we didn't laugh, we'd cry with what we have to deal with. Anyway... Oan I... ...get you something? - Like what? I don't know. I mean... I don't know. Ooffee? I don't want to put you out. It's no trouble. It takes two minutes. - It's just coffee. - I hate to impose. - Don't be ridiculous! - Actually, I would love a cup. Hey, you want rum cake? - I shouldn't. - Have some. - No, I can't. - Oome on! I'm trying to lose a couple micrograms. - Where? - These stripes don't help. You look great! I don't know if you know anything about fashion. Are you all right? No. He's making the tie in the cab as they're flying up Madison. He finally gets there. He runs up the steps into the church. The wedding is on. And he says, "Watermelon? I thought you said Guatemalan. Why would I marry a watermelon?" Is that a bee joke? That's the kind of stuff we do. Yeah, different. So, what are you gonna do, Barry? About work? I don't know. I want to do my part for the hive, but I can't do it the way they want. I know how you feel. - You do? - Sure. My parents wanted me to be a lawyer or a doctor, but I wanted to be a florist. - Really? - My only interest is flowers. Our new queen was just elected with that same campaign slogan. Anyway, if you look... There's my hive right there. See it? You're in Sheep Meadow! Yes! I'm right off the Turtle Pond! No way! I know that area. I lost a toe ring there once. - Why do girls put rings on their toes? - Why not? - It's like putting a hat on your knee. - Maybe I'll try that. - You all right, ma'am? - Oh, yeah. Fine. Just having two cups of coffee! Anyway, this has been great. Thanks for the coffee. Yeah, it's no trouble. Sorry I couldn't finish it. If I did, I'd be up the rest of my life. Are you...? Oan I take a piece of this with me? Sure! Here, have a crumb. - Thanks! - Yeah. All right. Well, then... I guess I'll see you around. Or not. OK, Barry. And thank you so much again... for before. Oh, that? That was nothing. Well, not nothing, but... Anyway... This can't possibly work. He's all set to go. We may as well try it. OK, Dave, pull the chute. - Sounds amazing. - It was amazing! It was the scariest, happiest moment of my life. Humans! I can't believe you were with humans! Giant, scary humans! What were they like? Huge and crazy. They talk crazy. They eat crazy giant things. They drive crazy. - Do they try and kill you, like on TV? - Some of them. But some of them don't. - How'd you get back? - Poodle. You did it, and I'm glad. You saw whatever you wanted to see. You had your "experience." Now you can pick out yourjob and be normal. - Well... - Well? Well, I met someone. You did? Was she Bee-ish? - A wasp?! Your parents will kill you! - No, no, no, not a wasp. - Spider? - I'm not attracted to spiders. I know it's the hottest thing, with the eight legs and all. I can't get by that face. So who is she? She's... human. No, no. That's a bee law. You wouldn't break a bee law. - Her name's Vanessa. - Oh, boy. She's so nice. And she's a florist! Oh, no! You're dating a human florist! We're not dating. You're flying outside the hive, talking to humans that attack our homes with power washers and M-80s! One-eighth a stick of dynamite! She saved my life! And she understands me. This is over! Eat this. This is not over! What was that? - They call it a crumb. - It was so stingin' stripey! And that's not what they eat. That's what falls off what they eat! - You know what a Oinnabon is? - No. It's bread and cinnamon and frosting. They heat it up... Sit down! ...really hot! - Listen to me! We are not them! We're us. There's us and there's them! Yes, but who can deny the heart that is yearning? There's no yearning. Stop yearning. Listen to me! You have got to start thinking bee, my friend. Thinking bee! - Thinking bee. - Thinking bee. Thinking bee! Thinking bee! Thinking bee! Thinking bee! There he is. He's in the pool. You know what your problem is, Barry? I gotta start thinking bee? How much longer will this go on? It's been three days! Why aren't you working? I've got a lot of big life decisions to think about. What life? You have no life! You have no job. You're barely a bee! Would it kill you to make a little honey? Barry, come out. Your father's talking to you. Martin, would you talk to him? Barry, I'm talking to you! You coming? Got everything? All set! Go ahead. I'll catch up. Don't be too long. Watch this! Vanessa! - We're still here. - I told you not to yell at him. He doesn't respond to yelling! - Then why yell at me? - Because you don't listen! I'm not listening to this. Sorry, I've gotta go. - Where are you going? - I'm meeting a friend. A girl? Is this why you can't decide? Bye. I just hope she's Bee-ish. They have a huge parade of flowers every year in Pasadena? To be in the Tournament of Roses, that's every florist's dream! Up on a float, surrounded by flowers, crowds cheering. A tournament. Do the roses compete in athletic events? No. All right, I've got one. How come you don't fly everywhere? It's exhausting. Why don't you run everywhere? It's faster. Yeah, OK, I see, I see. All right, your turn. TiVo. You can just freeze live TV? That's insane! You don't have that? We have Hivo, but it's a disease. It's a horrible, horrible disease. Oh, my. Dumb bees! You must want to sting all those jerks. We try not to sting. It's usually fatal for us. So you have to watch your temper. Very carefully. You kick a wall, take a walk, write an angry letter and throw it out. Work through it like any emotion: Anger, jealousy, lust. Oh, my goodness! Are you OK? Yeah. - What is wrong with you?! - It's a bug. He's not bothering anybody. Get out of here, you creep! What was that? A Pic 'N' Save circular? Yeah, it was. How did you know? It felt like about 10 pages. Seventy-five is pretty much our limit. You've really got that down to a science. - I lost a cousin to Italian Vogue. - I'll bet. What in the name of Mighty Hercules is this? How did this get here? Oute Bee, Golden Blossom, Ray Liotta Private Select? - Is he that actor? - I never heard of him. - Why is this here? - For people. We eat it. You don't have enough food of your own? - Well, yes. - How do you get it? - Bees make it. - I know who makes it! And it's hard to make it! There's heating, cooling, stirring. You need a whole Krelman thing! - It's organic. - It's our-ganic! It's just honey, Barry. Just what?! Bees don't know about this! This is stealing! A lot of stealing! You've taken our homes, schools, hospitals! This is all we have! And it's on sale?! I'm getting to the bottom of this. I'm getting to the bottom of all of this! Hey, Hector. - You almost done? - Almost. He is here. I sense it. Well, I guess I'll go home now and just leave this nice honey out, with no one around. You're busted, box boy! I knew I heard something. So you can talk! I can talk. And now you'll start talking! Where you getting the sweet stuff? Who's your supplier? I don't understand. I thought we were friends. The last thing we want to do is upset bees! You're too late! It's ours now! You, sir, have crossed the wrong sword! You, sir, will be lunch for my iguana, Ignacio! Where is the honey coming from? Tell me where! Honey Farms! It comes from Honey Farms! Orazy person! What horrible thing has happened here? These faces, they never knew what hit them. And now they're on the road to nowhere! Just keep still. What? You're not dead? Do I look dead? They will wipe anything that moves. Where you headed? To Honey Farms. I am onto something huge here. I'm going to Alaska. Moose blood, crazy stuff. Blows your head off! I'm going to Tacoma. - And you? - He really is dead. All right. Uh-oh! - What is that?! - Oh, no! - A wiper! Triple blade! - Triple blade? Jump on! It's your only chance, bee! Why does everything have to be so doggone clean?! How much do you people need to see?! Open your eyes! Stick your head out the window! From NPR News in Washington, I'm Oarl Kasell. But don't kill no more bugs! - Bee! - Moose blood guy!! - You hear something? - Like what? Like tiny screaming. Turn off the radio. Whassup, bee boy? Hey, Blood. Just a row of honey jars, as far as the eye could see. Wow! I assume wherever this truck goes is where they're getting it. I mean, that honey's ours. - Bees hang tight. - We're all jammed in. It's a close community. Not us, man. We on our own. Every mosquito on his own. - What if you get in trouble? - You a mosquito, you in trouble. Nobody likes us. They just smack. See a mosquito, smack, smack! At least you're out in the world. You must meet girls. Mosquito girls try to trade up, get with a moth, dragonfly. Mosquito girl don't want no mosquito. You got to be kidding me! Mooseblood's about to leave the building! So long, bee! - Hey, guys! - Mooseblood! I knew I'd catch y'all down here. Did you bring your crazy straw? We throw it in jars, slap a label on it, and it's pretty much pure profit. What is this place? A bee's got a brain the size of a pinhead. They are pinheads! Pinhead. - Oheck out the new smoker. - Oh, sweet. That's the one you want. The Thomas 3000! Smoker? Ninety puffs a minute, semi-automatic. Twice the nicotine, all the tar. A couple breaths of this knocks them right out. They make the honey, and we make the money. "They make the honey, and we make the money"? Oh, my! What's going on? Are you OK? Yeah. It doesn't last too long. Do you know you're in a fake hive with fake walls? Our queen was moved here. We had no choice. This is your queen? That's a man in women's clothes! That's a drag queen! What is this? Oh, no! There's hundreds of them! Bee honey. Our honey is being brazenly stolen on a massive scale! This is worse than anything bears have done! I intend to do something. Oh, Barry, stop. Who told you humans are taking our honey? That's a rumor. Do these look like rumors? That's a conspiracy theory. These are obviously doctored photos. How did you get mixed up in this? He's been talking to humans. - What? - Talking to humans?! He has a human girlfriend. And they make out! Make out? Barry! We do not. - You wish you could. - Whose side are you on? The bees! I dated a cricket once in San Antonio. Those crazy legs kept me up all night. Barry, this is what you want to do with your life? I want to do it for all our lives. Nobody works harder than bees! Dad, I remember you coming home so overworked your hands were still stirring. You couldn't stop. I remember that. What right do they have to our honey? We live on two cups a year. They put it in lip balm for no reason whatsoever! Even if it's true, what can one bee do? Sting them where it really hurts. In the face! The eye! - That would hurt. - No. Up the nose? That's a killer. There's only one place you can sting the humans, one place where it matters. Hive at Five, the hive's only full-hour action news source. No more bee beards! With Bob Bumble at the anchor desk. Weather with Storm Stinger. Sports with Buzz Larvi. And Jeanette Ohung. - Good evening. I'm Bob Bumble. - And I'm Jeanette Ohung. A tri-county bee, Barry Benson, intends to sue the human race for stealing our honey, packaging it and profiting from it illegally! Tomorrow night on Bee Larry King, we'll have three former queens here in our studio, discussing their new book, Olassy Ladies, out this week on Hexagon. Tonight we're talking to Barry Benson. Did you ever think, "I'm a kid from the hive. I can't do this"? Bees have never been afraid to change the world. What about Bee Oolumbus? Bee Gandhi? Bejesus? Where I'm from, we'd never sue humans. We were thinking of stickball or candy stores. How old are you? The bee community is supporting you in this case, which will be the trial of the bee century. You know, they have a Larry King in the human world too. It's a common name. Next week... He looks like you and has a show and suspenders and colored dots... Next week... Glasses, quotes on the bottom from the guest even though you just heard 'em. Bear Week next week! They're scary, hairy and here live. Always leans forward, pointy shoulders, squinty eyes, very Jewish. In tennis, you attack at the point of weakness! It was my grandmother, Ken. She's 81. Honey, her backhand's a joke! I'm not gonna take advantage of that? Quiet, please. Actual work going on here. - Is that that same bee? - Yes, it is! I'm helping him sue the human race. - Hello. - Hello, bee. This is Ken. Yeah, I remember you. Timberland, size ten and a half. Vibram sole, I believe. Why does he talk again? Listen, you better go 'cause we're really busy working. But it's our yogurt night! Bye-bye. Why is yogurt night so difficult?! You poor thing. You two have been at this for hours! Yes, and Adam here has been a huge help. - Frosting... - How many sugars? Just one. I try not to use the competition. So why are you helping me? Bees have good qualities. And it takes my mind off the shop. Instead of flowers, people are giving balloon bouquets now. Those are great, if you're three. And artificial flowers. - Oh, those just get me psychotic! - Yeah, me too. Bent stingers, pointless pollination. Bees must hate those fake things! Nothing worse than a daffodil that's had work done. Maybe this could make up for it a little bit. - This lawsuit's a pretty big deal. - I guess. You sure you want to go through with it? Am I sure? When I'm done with the humans, they won't be able to say, "Honey, I'm home," without paying a royalty! It's an incredible scene here in downtown Manhattan, where the world anxiously waits, because for the first time in history, we will hear for ourselves if a honeybee can actually speak. What have we gotten into here, Barry? It's pretty big, isn't it? I can't believe how many humans don't work during the day. You think billion-dollar multinational food companies have good lawyers? Everybody needs to stay behind the barricade. - What's the matter? - I don't know, I just got a chill. Well, if it isn't the bee team. You boys work on this? All rise! The Honorable Judge Bumbleton presiding. All right. Oase number 4475, Superior Oourt of New York, Barry Bee Benson v. the Honey Industry is now in session. Mr. Montgomery, you're representing the five food companies collectively? A privilege. Mr. Benson... you're representing all the bees of the world? I'm kidding. Yes, Your Honor, we're ready to proceed. Mr. Montgomery, your opening statement, please. Ladies and gentlemen of the jury, my grandmother was a simple woman. Born on a farm, she believed it was man's divine right to benefit from the bounty of nature God put before us. If we lived in the topsy-turvy world Mr. Benson imagines, just think of what would it mean. I would have to negotiate with the silkworm for the elastic in my britches! Talking bee! How do we know this isn't some sort of holographic motion-picture-capture Hollywood wizardry? They could be using laser beams! Robotics! Ventriloquism! Oloning! For all we know, he could be on steroids! Mr. Benson? Ladies and gentlemen, there's no trickery here. I'm just an ordinary bee. Honey's pretty important to me. It's important to all bees. We invented it! We make it. And we protect it with our lives. Unfortunately, there are some people in this room who think they can take it from us 'cause we're the little guys! I'm hoping that, after this is all over, you'll see how, by taking our honey, you not only take everything we have but everything we are! I wish he'd dress like that all the time. So nice! Oall your first witness. So, Mr. Klauss Vanderhayden of Honey Farms, big company you have. I suppose so. I see you also own Honeyburton and Honron! Yes, they provide beekeepers for our farms. Beekeeper. I find that to be a very
**Post 3:**

> Exclusive Brethren school accused of money laundering. Gareth Hales of Unispace, son of Exclusive Brethren (Plymouth Brethren Christian Church) leader Bruce D. Hales conspired with OneSchool Global to bilk charity funds and evade tax.   
  
[Get A Life Ep. 97 Tax Fraud, Embezzlement and the end of a Dynasty](https://www.youtube.com/watch?v=gn8X-REFPSQ)  


The OneSchool/Unispace scandal involves a complex scheme where a very large sum of money - estimated to be at least $80,000,000 - donated by members of the Plymouth Brethren Christian Church (PBCC) to the PBCC school charity (OneSchool Global) was stolen by Gareth and Charles Hales, sons of the PBCC's spiritual leader, Bruce D Hales. This scandal came to light when Steve Simmons, a former trustee of a OneSchool Global campus in Auckland, New Zealand, shared his story on social media.

Steve, who had served as a trustee for many years, possesses an abundance of emails and other records from that time to support his story. The transactions involved in the theft were deliberately complex and occurred over an extended period. Steve presented detailed evidence of these transactions, though the amount of information can make it challenging for the average listener to fully understand what happened and how.

This article is a summary of the entire scheme and will omit many details. It will be divided into four parts:

The game - how the theft was carried out

What happened in Auckland

Subsequent events at other campuses worldwide

The sale of Unispace and its replacement

**The game - how it was done**

To better understand the story, it's helpful to grasp the various players and processes involved in the abstract. There are three key players:

the Donor, who provides the money;

the Charity, the legitimate recipient of the money;

and the Operator, who manipulates the other players to steal as much money as possible for themselves.

The essential factor is that the Operator always remains in control. The game begins when the Operator establishes the Charity, which is officially registered and appears legitimate, convincing the donors to contribute funds. Initially, the money is used for its intended purpose, and the Charity achieves impressive results to showcase to the donors. This stage continues for several years, during which both the charity and the donations increase in size.

In the second stage of the game, the Charity accumulates enough income to fulfil all its objectives. However, this information is concealed from the Donor, who is subjected to ongoing marketing efforts to encourage them to donate more and more money. As a result, the Charity amasses significant cash reserves without any designated use for these funds.

The Operator now faces the challenge of transferring this surplus to their own pocket without arousing suspicion from the Donor or the government. This marks the third stage of the game. The Operator accomplishes this by establishing legitimate, for-profit businesses and selling goods and services to the Charity at significantly inflated prices. In order to compel the Charity trustees to purchase overpriced goods from his own businesses and control the Charity, the Operator must exert strong influence over the trustees. If the scheme succeeds, both the Operator and the Charity will possess a complete set of seemingly authentic documentation for all their transactions, which will be sufficient to satisfy tax inspectors. This practice is a well-known form of money laundering, where profits generated from criminal activities, such as fraud in this case, are disguised as legitimate trade transactions.

The fraudulent nature of the relationship between the parties involved can be demonstrated by three factors:

Conflicts of interest: The Operator wields control or influence over both the Charity and the Donor.

Single-source contracts: The Charity is always compelled to buy exclusively from the Operator, instead of obtaining competitive bids.

Cost-benefit analysis: The Charity incurs disproportionately high costs relative to the benefits it receives.

It should be noted that governments offer significant tax deductions for charitable contributions, making them a donor as well. Approximately 30% of the amount the Operator procures is stolen from the government and, consequently, from the general public.

**What happened in Auckland**

The PBCC schooling system was established in the 1990s and rapidly expanded globally under the guidance of PBCC leader Bruce Hales, providing a comprehensive curriculum for PBCC children. The operation of the schools was financed through a combination of tax-deductible donations from PBCC members and government subsidies. At the time of the incident in Auckland, there were approximately 9,000 children attending PBCC schools, with the largest numbers in the UK, New Zealand, Australia, Canada, and the USA.

For the sake of simplicity, I will refer to the PBCC school trust as OneSchool Global (OSG). During the events in Auckland, the local trust was known as the Hillsborough Education Trust, but all PBCC education trusts globally were rebranded as OneSchool while this ordeal was ongoing. The key players in Auckland were as follows:

The Donor: The Auckland brethren (PBCC members) and their businesses, as well as the NZ Government.

The Charity: OneSchool Global, the PBCC schooling system catering to Brethren children.

The Operator: Gareth and Charles Hales, who served as directors of the global construction company Unispace and were the sons of Bruce D Hales, the global PBCC leader. Local Unispace Directors were also in on the scheme.

In 2013, the Auckland OSG trustees received formal notification that their campus's senior school had been selected as one of three locations for piloting a new Learning Centre. This initiative originated from the PBCC's HQ in Sydney, Australia, where PBCC leader Bruce Hales and his sons reside. Unispace, which is Gareth and Charles Hales's business, would undertake the project. The trustees were informed that the estimated cost for the project would be around $320,000.

The message conveying this information came from Neville Simmons, who served as both a Unispace director, an OSG trustee, and the de facto leader of the Auckland congregation of the church. Notably, Neville requested that this message not be recorded in the trust minutes.

However, when Unispace presented the invoice at an early stage of the project, it amounted to $647,000 instead of the verbally-quoted $320,000 by Neville Simmons.

Steve Simmons was outraged and sought an independent estimate from a quantity surveyor for the work. The estimate came out to be only $180,000. Steve also reviewed the costs for individual line items on the Unispace invoice, such as carpet, chairs, and the firealarm system. He discovered that all of these were charged at approximately three times the commercial rate. For instance, the students' chairs were priced at about $1700 each.

As the trustee responsible for finance, Steve strongly objected, insisting that alternative quotes should be obtained. However, his objection was met with public humiliation. Neville and another trustee, Greg Mason, who was Neville's brother-in-law and fellow church leader, angrily dressed Steve down in front of the entire trust for a duration of about half an hour. They accused Steve of disloyalty to PBCC leadership and claimed that Bruce Hales, the global leader of PBCC, was furious.

Despite Steve's objections, his concerns were dismissed, and the other trustees aligned themselves with the directions from "Sydney". Unispace completed the work, and the charity paid them the full $650,000, diverting at least $450,000 of charitable funds, which were originally intended for the Brethrens' children, into the pockets of the Hales family and their business associates.

A couple of years later, a new directive regarding education was issued by PBCC HQ in Sydney. The junior schools were required to adopt an open plan, moving away from the individual classroom system. The project involved removing walls, installing structural beams, and converting the space into a single large room. Unispace was once again assigned the task, requesting a payment of approximately $35,000 from the Auckland trust solely for providing a quotation. The quotation arrived, totalling around $1.3 million, along with a demand for approval within two days.

Since the Trust had depleted its funds by this point, Steve managed to convince the trustees that they should undertake the work themselves, utilizing volunteer labour from the local congregation. The project was successfully completed for a cost of about $35,000, the same amount they had paid Unispace merely for the quotation fee.The quotation fee was never refunded.

Shortly after this incident, Steve was forced off the school trust. He continued to raise concerns with leading Auckland brethren about Unispace's exorbitant overcharging and misappropriation of charitable funds. Consequently, Steve was "shut-up" in 2020 and "withdrawn from" in 2022. These two stages denote excommunication from the PBCC. "Shut-up" meant that Steve could no longer attend Church services and had to live, eat, and sleep separately from his wife and children. "Withdrawn from" entailed a complete physical and spiritual severance from the church and its members, with no social contact permitted. Naturally, these circumstances led to complete estrangement from parents and children and often resulted in marital divorce.

Steve was not the only PBCC member in New Zealand concerned about the monopoly of Unispace. Mike Powell, a highly-qualified PBCC accountant from Hamilton, apparently also expressed serious concerns about conflicts of interest and breaches of tax and charity law in the relationship between Unispace and OneSchool in a letter to Bruce Hales.

However, Mike was forced to withdraw his email under threat of excommunication. The Hamilton congregation was called together for a special administrative "Assembly Meeting" to withdraw from him. The congregation waited for a long time while the PBCC priests pressured Mike to sign a retraction of his allegations and a non-disclosure agreement (NDA). Eventually, Mike caved under the pressure, the congregation was dismissed, and Mike remained in the PBCC, unable to speak out.

The Unispace grift extended beyond school campuses. In an incident in Auckland, the New Zealand national headquarters for OneSchool Global rented a whole floor of an upscale office building owned by church leader Peter Bishop. During the office setup, Neville Simmons had the partially completed installation ripped out and replaced by a very expensive and luxurious interior by Unispace, of which he was a director. Both the cost of the Unispace fit-out and the ongoing rent to Peter Bishop were charged to the charity, OSG, and paid for with funds donated by the brethren for children's education.

**What happened globally**

The Auckland Learning Centre and two similar learning centers in NSW, Australia - the Illawarra Campus in Darkes Forest and the Sydney campus in Oatlands, were pilot projects for a new teaching approach in OSG. The plan was for senior students to spend much of their time working autonomously in an open-plan setting rather than being taught by traditional teachers in a classroom. To implement this new system worldwide, every OSG campus would need to construct a new Learning Centre with special desks, furniture, and computer systems.

The two other pilot learning centers in Australia were also constructed by Unispace at a cost similar to the Auckland project. Once these pilot projects were deemed successful, all 120 or so OSG campuses globally were instructed to follow suit. While the exact details of every case are unknown, according to Steve Simmons, the following occurred:

"A month or two after our Learning Centre was finished, Bruce Hales travelled to the UK for several big education strategy meetings. Senior trustees from around the world were present. During one of these meetings, Bruce explicitly stated that every Learning Centre project at each campus worldwide should be exclusively assigned to Unispace.Essentially, he ordered his followers from across the globe to give his company (or perhaps his sons' company) millions of dollars' worth of projects at significantly inflated prices. Then he returned to the UK a few months later for another major education meeting. At the meeting, a few senior trustees from our campus were present, although I can't recall the specific names. Whoever it was, they provided our trust with an overview of what had been discussed during their trip. I distinctly remember them sharing about some brothers in the UK who had dared to seek quotes from companies other than Unispace.It's worth noting that Unispace's pricing was exorbitant, and it would have been much easier to find much cheaper options elsewhere. Nevertheless, the point that was emphasized to our trust, possibly directed at me, was that "the brothers who did that could consider themselves lucky to still be in fellowship."

If we multiply the $450,000 that Unispace took from each pilot project by the 120 campuses worldwide, the total comes to $54 million. This figure is solely for the Learning Centres. If we also consider the Junior School projects, the national HQ office renovations, and the potential construction of entirely new campuses in some locations, the overall amount of money channelled from the brethren, through OneSchool Global and Unispace, into the pockets of the Hales family, could easily reach around $80 million. All of this occurred in a span of approximately six years, from 2014 to 2020.

To put this into perspective, stealing $80 million is equivalent to taking $1,450 from each PBCC member, $250,000 from each congregation, or nearly $8,500 from every OSG student.

**How Unispace was sold and what replaced it.**

In March 2021, Unispace was sold to PAG Asia Capital for $300 million. The value of a business is typically calculated as a multiple of its annual profits. Therefore, the six years of highly profitable work for OneSchool Global that preceded the sale likely greatly increased Unispace's sale price. However, it is unclear whether PAG Asia Capital was aware that the loyalty of OneSchool Global to Unispace was coerced rather than a free market decision. It seems probable that Gareth and Charles Hales significantly multiplied their gains through what is known as a "pump and dump" scheme, where the value of a business is artificially inflated through contrived transactions before its sale.

One division of Unispace, Unispace Health (later rebranded as Santé Global), remained under the ownership of the Hales brothers and became the largest recipient of Covid PPE contracts from the UK Conservative government's illegal VIP lane single-source procurement process. This totalled £680 million (NZ$1.4 billion). An investigation conducted by anti-corruption whistleblower the Good Law Project revealed that the Unispace contracts were facilitated by Cabinet Minister Michael Gove. Freedom of Information requests showed that a "founder" of Unispace, whose name has been redacted, spoke to Gove and subsequently sent an email outlining their offer later that day, expressing prayers for Gove and the Conservative party during a challenging time.

The sale of Unispace may appear to have left the Hales family without a means to exploit OneSchool Global's income. However, this is not the case. When the office fitout division of Unispace was sold, many employees transitioned to other large businesses within the Brethren network operating in the same market. A number of these businesses are connected to the Hales family and are now competing with Unispace, potentially violating the terms of the sale. These businesses can continue to coercively obtain contracts from OneSchool Global and funnel the proceeds back to the leading Hales family within the Plymouth Brethren Christian Church (PBCC). Furthermore, by distributing the operation across multiple businesses, the common factor of Hales control is less apparent, making the scam more difficult to detect.

Another important point to note is that OneSchool Global is just one of many entities from which the Hales can extract profits within the PBCC "ecosystem". A significant portion of funding for OneSchool Global flows through the Universal Business Team (UBT), the finance arm of the PBCC. UBT provides phones, computers, accounting, consultancy, legal insurance, and conference services for all 55,000 brethren and their 3,000 businesses globally. Additionally, there is Campus&Co, a global members-only supermarket chain owned by UBT, which aims to capture the majority of PBCC members' and businesses' grocery spending. Furthermore, there is OneMedifund, which provides health insurance for all members and various other initiatives that purportedly donate all profits to OneSchool Global.

Each of these enterprises can be exploited using the same process as was employed at the OneSchool Global Auckland campus. They are all ultimately controlled by the Hales family, and if instructed to purchase goods and services from a specified supplier, they would do so unquestioningly. It is highly likely that the Hales family receives kickbacks from phone and computer manufacturers for directing captive customers their way.

One intriguing aspect is that Campus&Co is specifically mandated to purchase all its wines and spirits from a business called Vendimia, a for-profit corporation owned by prominent PBCC members. The purpose behind this strict limitation is unclear, unless it is intended to divert profits to the PBCC leadership.
**Post 4:**

> S my roommate spying on me ? Hey guys, I'm not usually the type of person to ask for advice on the internet but I don’t really know what to do right now. I don't want to be like one of those recipe pages where they tell you their entire life story before actually getting to the recipe, but in order to help me I think it's important to get some background on my situation.

I met my roommate, we’ll call them Cameron, online. I know what you're thinking, I had kinda signed up for some weird shit rooming with some rando. The reason I resorted to finding someone online instead of asking a friend is that a
A) I don't have a lot of friends, and B) The few friends I do have either want to live alone or live with their partner/family so I didn't really have a choice and C) I am not really in contact with my family right now. I would live on my own if I could, but I am currently working as an intern and barely make enough to scrape by. 

This is how I ended up on the “looking for roommates in their 20s” facebook page. Cameron messaged me first, and I thought it was really weird that they didn't have any facebook friends nor a profile picture, but I thought the polite thing to do was respond. However, we got to talking and it seemed like we had a lot in common - we both are okay with clutter as long as it is not gross (i.e moldy food or poorly unkempt bathrooms/hygiene), and loved cats, horror movies, and time to recharge on our own. 

Cameron told me that they specifically made their facebook page to find a roommate and isnt really interested in having a social media presence which is why there is very little info on their page. I can respect that, sometimes I wish I never got my first facebook account at 13 and relished unrestricted internet access. 

Cameron said that they were just moving into the cities this year from a small farming town up north and was a little nervous about living in the cities for the first time. After talking more, we decided to meet up in person. We agreed to meet at a small diner halfway between their current home and mine, which still ended up being a 2 hour drive. They had gotten there first, and when they stood up to greet me I was stunned as they did not look like I expected them to. Cameron was tall, muscular, and incredibly attractive. Their pale blue eyes seemed to glow in the dark and their teeth lined up as straight and uniform as a military cemetery. 

I found myself jealous, as my eyes were dark brown (and not the pretty dark brown like puppy dog eyes, like sewer water brown) and never got braces when I should have. After I got over my initial shock, I bustled my way over to try to cover up my reaction. We shook hands and sat across from one another in an old booth that seemed incredibly worn but still felt welcoming. 

We discussed our desired living situation: how many bathrooms, did we need a dishwasher or in unit laundry, was AC necessary, are we okay with pets, rent budget, and neighborhood we wanted to live in. Cameron appeared to be the perfect roommate, agreeing with everything I said I was looking for in an apartment. I readily accepted their offer to move in together.

We found the perfect apartment and started moving in 2 months after we first started talking on facebook. Neither of us had much so we were able to move all of our things ourselves without having to rent a moving van. Our layout looked something like this: our bedrooms were linked with a jack and jill bathroom that had plenty of space for 2 people to keep their things and get ready at the same time. The front door opened up into the living room with the closet containing the washer and dryer to the immediate right. The kitchen was to the left and there was a small island between the kitchen and living room. I already had 2 cats, Seymour and Audrey, that Cameron was very happy to live with. At first glance, it seemed like we were going to be great roommates.

Okay, sorry for rambling. I just want to show that everything was normal at first. Our lives intertwined seamlessly - I left for work before they got up, and they got home when I was starting to get ready for bed. We would occasionally eat dinner together on the weekends and play video games and watch movies and stuff, and sometimes go out for a beer. That is how it was for the first 5 months or so - until I noticed the hole in my bedroom wall. I saw it after I started to hang things up on that wall and noticed a small chip in the paint. I inspected it, as I did not notice it when we were filling out the form with all the damages from before we moved in so they wouldn't be counted when we got our security deposit back. 

As I looked at it more closely, I saw something that reflected off of my lamp for just a second. I got my phone to use as a flashlight and spotted a small, round, black object that was no bigger than a peppercorn. The hole was too small to reach in with my fingers, so I started searching for things small enough to grab whatever it was. I tried everything I could think of: tweezers, tooth pics, tape, pliers, cotton swabs, you name it. Eventually I gave up and just went to bed.

I couldn't sleep that night, all I could do was stare at the ceiling and think about what it could be. After a few hours of tossing and turning - i decided that if i couldn't get it out, i have to make the hole bigger. I wasn't too worried about the security deposit since the hole was already made, what difference does it make if it's just a little bit bigger?

I didn't have any tools, so I dug around until I found a small pocket knife that my grandpa got me as a souvenir from Alaska. It said, in all caps, “I LOVE MY GRANDPA!!” I slowly made the hole bigger until it was enough to pull the object out with my tweezers. I celebrated as soon as i got it out, until i realized what it was.

It was a pin camera.

I only recognized it because I worked as a tour guide for a spy museum one summer. My mind went blank before the sirens in my head started going off. What the fuck was this? Who put it here? Is it the landlord watching us? Was there a person living in the ceiling that only came out when we were gone? My brain reeled with possibilities, mostly related to situations in different horror movies I had seen (I had just watched 13 Cameras last week), but one thing was certain: I had to search the apartment for other cameras. I just have to wait until Cameron is gone.

05/11/2024

Cameron is gone for the weekend visiting family. We had previously discussed that it would be okay to quickly run into each other's rooms if needed, like if the cats were going crazy or we needed to borrow something. I hadn't had a reason to go into their room yet, but I had to see if there was a camera in their room too. I didn't want to freak them out without evidence. I opened their door and was a little surprised at the lack of furniture. 

Cameron had a twin size mattress on the floor without a bed frame, a small desk, and a wooden cutting board that they must have been using as a bedside table. There was nothing on any of the walls, no lamps, and no TV or computer that I could see. It was the polar opposite of my room (it looked like it was straight out of howl's moving castle). I was relieved that I didn't have to move a ton of furniture or take things off of the wall though. I started on the wall next to the bathroom where mine was and when I didn't find it I started searching up and down every wall.

Nothing.

As soon as I started to walk out, I realized I hadn't looked in the closet yet. I went over and pulled open the sliding door and froze. The clothes in their closet were almost the same as mine. I closed the door and left. I was a little weirded out, but we had a similar job and we never saw each other wearing anything other than pajamas or sweatpants so I chalked it up to a coincidence. 

Anyways, I left their room and softly closed the door. I leaned against it and took a deep breath - half relieved and half terrified. It was good that Cameron wasn’t being targeted, but that meant that someone was watching me specifically.

05/21/2024

After our last conversation, I'm pretty sure it's Cameron.

The next saturday that we were both home, I decided to bring it up. I brought them into my room and showed them the hole and the camera. They said, “what the fuck dude, thats messed up” but his face showed little emotion. I suspected they were just in shock, as I was when I found it. Then I told them I had gone into their room. They became rigid and slowly turned their whole body at me, as if they were one solid being without joints or limbs. They stared at me and I felt their eyes in my soul as they blazed with an emotion I can't name - fear? Hatred? Confusion? I wasn't sure.

“You went in my room?” they asked coldly. I guess it was more of a statement than a question, but I answered them anyways

“Yeah, I wanted to make sure that there were no cameras in your room so you didn't freak out and tear apart your room trying to find one. I didn't find any though, I guess they're only interested in me'' I laughed, trying to avoid the fear that was striking its way through my veins.

“You. went. In. my. Room.”

“Uh.. yeah?” I was confused. This was something clearly stated in our roommate agreement that we could go into each other's rooms if necessary.

“Did you see anything weird in there”

“I just told you I didn't see any cameras. Don't worry, your room seems normal”

“Okay” silence filled the space between us. I stood under Camerons gaze, feeling as though I were being dissected. Then, Cameron flashed their 100 watt smile.

“Good,” they said. “I can help you search your room and the rest of the place for other cameras?” 

We searched our apartment top to bottom. Nothing. While I was still on edge, I felt better knowing it was the only one. Maybe it was left over from a previous resident freak? I decided to plug the camera into my computer to see if there was any footage on it. Thankfully, it seemed like it was really outdated and had no data on it. I threw the camera away, but decided it would be a good idea to put a small camera in my room to see if anyone strange came in or out. I didn't think to tell Cameron about it since it was my room, and it was a good thing I didn't.

06/13/2024

After a few weeks of recording, I decided to review the footage. It was mostly normal stuff: me on my computer, sleeping, cleaning, etc. that is, until the second week.

The door to my room creaked open slowly and a large figure began to walk through the darkness. My breath refused to come out of my lungs as my heart pulsated. When the light turned on, I felt like I was having a heart attack.

It was Cameron.

They started by doing a quick scan of the room before opening my closet. They didn’t do anything, just stared at it as if he was taking pictures with his mind. They did this all around my room, spending the most time looking at the pictures of me and my friends/family on my desk. After a few minutes, they turned off the light and quietly left my room.

After that, I really didn’t know what to do or how to interact with them. I tried to keep it cool but I could tell that I was acting weird, they didn’t say anything about it and went on as usual. I decided to do some background research on them (something I should have done in the first place).

I started easy by googling their name. I knew they didn’t have a social media presence but I still expected something to come up: a LinkedIn profile, a news article, a mention in a dean’s list or something - but absolutely nothing came up. I thought maybe I didn’t spell their last name right, so I decided to look at their ID when they were in the shower. I quietly went to the basket in the entryway where we keep our wallets and keys and slid the ID out of its pocket and froze.

Cameron’s name was not the one printed on the ID.

The picture was definitely them, the blonde hair and wide smile - but the eye color description said BRO, not BLU. I looked closely at the picture and realized that instead of their piercing blue eyes, it looked like the person in the picture had brown eyes. I quickly took a picture of their ID and put everything back as I had found it and escaped to my room.

I decided to google the name on the ID. Sure enough, a fair amount of information popped up. Whoever this person is seems to be a carefree and easy going person with lots of pictures of them with friends and family. I sighed in relief when I clicked on the facebook profile as it was public. I started from the bottom and scrolled through their posts. Most of it was the normal stuff: happy birthday wishes, memories from years prior, pictures of them at events and stuff.

I couldn’t help but notice their eyes. Again, they were not the piercing blue that I have come to know - they were brown, like a puppy staring at you with unconditional admiration. I didn’t think that Cameron wore contacts and I never saw any in our bathroom.

The final post was dated 2 years ago. It said: 

“Can’t wait to move into my new apartment! My new roommate Cameron is the best!!”

Posted with the update was a picture of the person with the brown eyes and another person that I did not recognize at first, until I noticed their piercing blue eyes.

Cameron’s eyes.

I have no words to describe what I was feeling at that moment. My heart sank so deep into my body I thought I would shit it out. I closed my laptop and stared at the wall, my thoughts coursing through my head so fast that it felt like I wasn’t thinking at all.

I decided to just go to bed.

06/15/2024

i usually try to write these a few days after stuff happens but i am not even going to have time to proofread this because i am typing as afast as i can cameron is outside my door right now banging on it to getty in i just want to document this and post it because i think i know what is going to happen to me

I opened myd oor a few minutes ago and saw myself standing outside but this copy fo me had those piercing blue eyes that i have become to fear and i knew in an instant that i had to document my experience as fast as possible because i dont know howm uch time i have left if you are seeing this tell my friend s i ov90e them and that whatever it looks like i am dojing in the future is not me and to stay awya i wouldnt wish this on lanoyone 


Comments:

UPDATE:
Hi, I know it has been a while since I posted this and I have to say I am very embarrassed about it. Turns out our apartment had a carbon monoxide leak and I was hallucinating. I am out of the hospital now, thankfully Cameron was at work when this happened and came home to find me passed out. I am not very internet savvy so I am not sure how to edit or delete this post but I did figure out how to comment.I just wanted to update and reassure you all that I am OK! I have decided to move back to my hometown to spend time with my family to take my mind off of things. Cameron is sad but understanding and is looking for a roommate. Attached below is a screenshot of their facebook if you are in the area and looking for a roommate!

*Image Description: screenshot of a facebook profile. The profile picture is of a man in his late 20’s with messy brown hair, slightly crooked teeth, and pale blue eyes. The account has no friends or posts*
**Post 5:**

> Just a father… TLDR: I ended a 9 year relationship because my partner was never present and had other priorities, now she’s being vindictive and manipulative leaving our son left in the middle of all her games and behaviour - all whilst I’m still being forced to fund her lavish lifestyle!

I’d like to start off by saying, this is my first post - I’ve been reluctant to post on here out of fear of backlash, adverse opinions but after reading so many other posts on the topic of the fathers side of child arrangement orders, it seems to be a recurring theme that fathers are almost always worse off after a relationship ends with a child involved (I'm not saying there aren't exceptions to this) and I’m just so confused!

I don’t know what I’m expecting as an outcome of posting this, ideally some sort of uprising I suppose, whereby the way that the court process for child arrangements actually provides a fairer outcome with the child’s best interests in mind - but albeit that seems a bit farfetched.

My main bug bear with my situation, is all these female “equal rights” advocates screaming about equality for things such as equal pay etc but when it comes to “parental rights”, I never hear advocates saying fathers should have an equal involvement in a child’s life.

Anyhow, to the point (key highlights/background):

+ Relationship started in 2014. We were in our early 20's.

+ Our son was born in December, 2015. 

+ Relationship was rocky throughout, common themes:

Spending all her time up the yard (horses)

Buying expensive sports cars (After agreeing we’d save for a house)

Claiming benefits (UC) when she shouldn’t have been. 

+ We went to counselling several times over the years for the above and other things. I attended counselling separately for some issues of my own (childhood trauma).

+ Relationship ended in 2023 for many reasons, I won’t go in to a massive amount of detail but she was never around, I looked after the kids (our son, and her daughter), I cooked and I cleaned the house. 

+ Following the breakup, she refused to speak to me or to allow me to have any contact with my son, simply claiming “he doesn’t want to” until I’d submitted a C100 to the courts for a child arrangements order after she refused to attend mediation whilst I continued paying her child support.

+ Fast forward to April, when the first hearing was due and she’d conveniently booked a holiday so couldn’t attend. A new hearing was then scheduled for June. 

+ In the June hearing, I’d expressed concerns relating to our sons safety and wellbeing as he’d been under the influence of marijuana and around people attempting to commit suicide whilst in her care, I didn’t say these things to make a point but because I was asked if I had any concerns, to which I’d answered honestly. An interim order was granted then and there allowing me to see my son again, on the basis of every Wednesday evening, and every other weekend, then half of all school holidays.

+ The next time I go to collect my son from school, he comes out acting very strange, almost refusing to make eye contact with me, then says “My mum told me to tell you I don’t want to see you today.”.  

+ Things continued like this for a while whilst the court hearing was ongoing. To confirm, she advised the courts that she had no concerns or reasons for me and my son to not spend time together or see each other, and to me it seems ridiculous that we’ve ended up in this situation after I decided to end a relationship on the basis that she spent no time around me or the kids. She just turned on the waterworks crying in the court room saying she just wants what’s best for him, as if I don’t?

+ During this period, my ex partner was also communicating with my mother, arranging for her to see our son. I was also ensuring that my mother was spending time with him. By the time of the final hearing it had equated to me and my mother spending 49% of his time with him, and him spending the other 51% of his time with his mother. 

+ During this period I continued to pay child maintenance, with the exception of January 2024 when I stated I was refusing to pay until she agreed to let me see our son or attend mediation to progress things (I don’t take pride in this, however we weren’t getting anywhere fast and the court were dragging their heels and it felt like the only leverage I had at the time on the situation). 

+ The final hearing occurred in October 2024, it seemed like a no brainer to me. I was asking for a 50/50 split, I’d purchased a house within a 20 minute drive of my ex partner and our son’s school. He’s never missed a day of school whilst in my care. I have a well paying job. We’re almost always out of the house doing fun activities, and my life is (and always has been) entirely focused around my son. He has always been extremely happy child and rarely complains.

+ Interim order stated that I would have my Son every Wednesday, as well as every other Weekend. I appreciate that this is more than some fathers get, but prior to the relationship ending this woman was never there so it felt like a real kick in the teeth, I was only asking for an additional night in the week.

+ My ex made out in the court hearing that I wasn’t facilitating any type of relationship between my mother and our son, and that it was all her doing. That she was having to let her (My mother) see him on her (My Ex's) time. Anyhow, the final outcome arrived and they dictated that things should “stay as they are”. It felt, and still feels very surreal.

+ The courtroom was full of magistrates who were all female, and I'm not saying that the verdict was solely based on them sympathising with the crying mother who's been dragged in to court to defend her rights to keep her son against her evil ex partner (me), but it definitely felt unfair that my ex partner had already been to court for the same reason with her daughter (refusing to let her ex see her also) so she understood the process and what to say.

+ We’ve had a number of issues since then, where I get told “he doesn’t want to see you” then I’ll arrive to a 9 year old child having a tantrum as if he’s been told I’m going to murder him if he comes with me. He's my little buddy, we've done everything together since he was born, we've built cars and travelled around the UK in them including the North Coast of Scotland (NC500, if you haven't done it already - then you need to!), gone fishing all over the solent, and do all the typical things you'd expect a father and son to do. We have an amazing relationship - but it's being tainted by the lies that his mother is still telling him to try and convince him that I'm the villain in this situation.

+ His sister (my ex partners daughter from a previous relationship) refuses to speak to me and states she hates me. Her father reached out, asking how I was and what had happened, as he was confused about why I’d suddenly abandoned his daughter. I explained that wasn’t the case and that I was actively trying to include her by inviting her out to dinner and for days out, but my Ex was filling her head with rubbish making out that I'd abandoned them all and didn't want anything to do with any of them.

+ Christmas 2024. The court order stated our son would wake up with one parent, then be collected by the other parent at 12PM on Christmas Day. I woke up with hope, but ended the day in tears. I dressed up as Santa to go and collect my son, I had all his presents ready under the tree, excited to spend the day with him. When I arrived, she (my ex partner) answered the door and scoffed at my outfit, and stated “he doesn’t want to come with you”, I asked to speak with him, and I could hear him screaming in the background. We spoke and I asked him calmly why he didn’t want to come, and he said “I just don’t want to”. I had flashbacks to when I was with my ex partner and would find her filling her daughters head with rubbish, saying things like "You don't want to go to your Dads, because you'll miss me too much won't you?".

+ Things don't appear to be getting much better, I've been sticking to the routine and my son has asked to stay with me longer on several occasions now, and when possible we'll call his mum up and ler her know he wants to stay - she's agreed to this twice since October '24.

+ We've booked to go away for this Easter holiday, to which she agreed to in January. After speaking this morning I fear that we're going to have a repeat of the Christmas incident due to her saying "I'll see how he feels" in response to me saying "I'll pick him up on Monday". Normally, someone would respond "Sure" or "Not a problem" opposed to pre-emptively thinking "He won't want to come". He's currently very excited and we've even arranged for him to bring a friend with us for the weekend, so it seems bizarre to imagine that he wouldn't want to go and spend a weekend away playing on the beach digging up fossils (Dorset).



Questions:

1) Why is the UK "justice" system still so backwards when it comes to child arrangement orders?

2) Why do mothers automatically get full access to a child, then fathers have to claw their way back in to a child's life slowly and painfully?

3) Why do child maintenance payments not count towards a mothers taxable income?
            My Ex is claiming benefits, has a new partner who earns an income, has a full time job herself and receives child maintenance from two fathers whilst driving around in her sports car worth £20k, has a horse lorry and three horses, All whilst I'm paying her 11% of my pre-tax salary, trying to pay for a mortgage so my son has a bedroom to live in and driving around in a 10 year old car.

4) Why was a verdict given, with no explanation? They simply stated that the verdict was "things should stay as they are", my presumption is that they came to this conclusion based on the following:
            A) My ex saying our son doesn't want to see me.
            B) Fear of my ex attempting to commit suicide again, after displaying her emotions in the court room.
            C) The lie my ex told, advising she was the only one facilitating a relationship with my paternal mother.



The entire situation has been immensely infuriating to be involved in, I'm a straightforward thinker and in my opinion the situation didn't pan out with my son's best interests in mind at all, he now spends majority of his time with his mother playing on his tablet, sat in his bedroom on his own, in the care of others (god forbid, I'd be allowed to have him any extra time) or up the yard whilst she tends to her horses. Her motives are solely based on ensuring that she continues to receive child maintenance payments.

I think the court system needs a serious review of the process, ideally whereby both parents are granted 50% care wherever possible by default, excluding abuse claims or where excessive travel would cause disruption to the child's schedule, and or lifestyle - but in this instance none of that was the case, in fact I'd argue the opposite that by granting his mother to have him more is having an adverse effect on his lifestyle and behaviour, as he now gets passed around more between her friends and family whilst she tends to her other "priorities". She now has full control over the situation and dictates where he goes, when and even who with telling me that I’m “just his father” implying I don’t have the same rights as she does whilst making decisions in his life, which feels to be strengthened by the outcome of the court case. 

This post is more of a rant / coping mechanism on the situation, and expressing my frustration about the fact that I spent several years being unhappy with my living situation in hopes things would improve before finally deciding to leave on the basis that I didn't think that the kids should have to grow up believing that what we (me and my ex) had was what a healthy relationship should look like, I knew things would never end amicably between us after witnessing the way she acted towards her ex partner - but I didn't expect her to act so manipulatively towards our son, who has noticeably been acting more withdrawn since the entire situation has unfolded.

Please don't interrupt the above as me implying that I'm Mr.Perfect, or that the relationship ended purely because of her actions, I spent several years attending counselling trying to work on myself, and now I'm trying not to dwell on the situation too much and just focus on the time I do still get to spend with my son, I already know that when he grows older he'll have a better understanding of manipulative behaviour - I just wonder at what cost?

I appreciate there was no outcome from the court case, where everyone would be happy. I equally appreciate that some time with my son is better than no time. In a world where everyone claims there are so many "deadbeat" Dads -  I'd like to think that everyone is as keen as I am to see a positive change in how the system works, yet it seems that the one's who are trying seem to get sh!t on the worst.

If you've managed to read this far, then kudos to you and thank you for your time.
## Topic 4: Data Privacy & Regulation
**Post 1:**

> Independents may build on Australia’s history of hung parliaments, if they can survive the campaign blues **Independents may build on Australia’s history of hung parliaments, if they can survive the campaign blues** 

**Joshua Black, Visitor, School of History, Australian National University, Published: April 28, 2025 6.05am AEST**

Major parties used to easily dismiss the rare politician who stood alone in parliament. These MPs could be written off as isolated idealists, and the press could condescend to them as noble, naïve and unlikely to succeed.

In November 1930, when independent country MP Harold Glowrey chose to sit on the crossbench of the Victorian parliament while his few peers joined the new United Country Party, the local newspapers [emphasised](https://trove.nla.gov.au/newspaper/article/269650721?searchTerm=Glowrey) that he could not “become a cabinet minister” or “have a say” in making policy from the sidelines. (As if he wasn’t aware.) Australia was a place where, according to the scribes at The Ouyen Mail, “very few constituencies were prepared to elect independent men”.

Things are rather different now. Lifelong loyalty to a single party has become a rarer thing among voters, with the Australian Election Study [showing](https://australianelectionstudy.org/wp-content/uploads/The-2022-Australian-Federal-Election-Results-from-the-Australian-Election-Study.pdf) fewer than four in ten voters give their first preference vote to the same party at each election. It was more than seven in ten back in 1967.

Voters have gravitated towards alternatives to the two major parties. A new interactive data tool from the ABC shows just how much [more competitive](https://www.abc.net.au/news/2025-04-24/election-data-rise-independents-major-party-drift/105144918) federal elections have become. Australians are now [world leaders](https://australiainstitute.org.au/post/election-entree-australia-is-a-world-leader-in-electing-independent-mps/) in sending independents to represent them in state and federal parliaments.

And who could call the independents of the recent past naïve? Independent MPs held the balance of power in [New South Wales](https://insidestory.org.au/close-enough-could-be-good-enough/) in the early 1990s, and in [Victoria](https://insidestory.org.au/victorias-unexpected-minority/) later that decade. Both parliaments saw substantive reforms and improved parliamentary processes.

**A strong track record**

At the federal level, a lineage of independents such as Ted Mack, Peter Andren, Zali Steggall, Cathy McGowan and her successor in Indi Helen Haines have all found new ways to give voice to their community in parliament. Voters, especially in rural electorates and formerly “safe” seats, have been attracted to candidates who promise to “do politics differently”, as McGowan so often [puts it](https://pathwaystopolitics.org.au/knowledge-hub/doing-politics-differently-with-cathy-mcgowan/).

There are dozens of candidates making that promise at this election. At least [129 candidates](https://kevinbonham.blogspot.com/2025/04/federal-2025-classifying-independents.html) are listed on House of Representatives ballot papers as independent or unaffiliated candidates in 88 seats. That’s almost twice as many independent candidates than in the [2013 election](https://results.aec.gov.au/17496/Website/HouseCandidates-17496-NAT.htm) for the lower house. Around 35 of these are community independent candidates. A further 28 people are running as independents or ungrouped candidates in [Senate races](https://www.aec.gov.au/media/2025/04-12.htm).

So who are the independent candidates, and what role might they play after May 3?

**Who are the independent candidates?**

For a start, around a third of all independent candidates for House of Representatives seats are women. Among the “community independent” candidates (commonly referred to as “teals”), it’s closer to four out of five.

This is entirely in keeping with the role daring women have played as the strongest custodians of non-party politics in Australia over the past [120-odd years](https://www.auswhn.com.au/blog/women-federal-politics/).

Most of the women on ballot papers this year are professionals and public figures. Nicolette Boele, candidate for Bradfield, NSW, is a former consultant and clean energy financier who came close to unseating cabinet minister Paul Fletcher in 2022. In the seat of Calare, also in NSW, candidate Kate Hook [describes](https://www.katehook.com.au/who_is_kate) herself as “a professional working mum” and “small farmer” with an interest in regional development and renewable energy. Caz Heise, candidate for Cowper (NSW) is a healthcare expert who carved a sizeable chunk out of the National Party vote in 2022. Independent candidate for Groom (Queensland) Suzie Holt is a social worker by training who finished second at the last election. Berowra’s Tina Brown is a local [magazine publisher](https://www.northshorelorikeet.com.au/p/who-is-tina-brown-the-independent-candidate-for-berowra) with deep roots in Sydney’s Hills District.

Who are the dozens on men putting themselves forward? Many are former mayors and councillors running for parliament while the opportunity presents itself. There are a small but noteworthy coterie of men running on a specifically Muslim platform, some of whom are running with the support of the Muslim Votes Matter [organisation](https://www.muslimvotesmatter.com.au/about-us).

Of the few “teal” men, the most competitive by far is Alex Dyson, a third-time candidate in the western Victorian seat of Wannon, currently held by Dan Tehan, shadow minister for immigration and citizenship.

A former Triple-J presenter and comedian with a “side-hustle” as an [Uber driver](https://www.sbs.com.au/news/article/a-former-triple-j-host-is-hoping-his-uber-side-hustle-will-help-him-unseat-a-liberal-mp/3q6yzclov), Dyson will hope to benefit from his positioning at the [top of the ballot paper](https://www.standard.net.au/story/8940473/independent-alex-dyson-tops-wannon-ballot-for-third-run/) for Wannon.

**Crossbench contenders**

Most of the women who swept into parliament in 2022 are campaigning to retain their seats. Dai Le in Fowler, Sophie Scamps in Mackellar, Allegra Spender in Wentworth, Zoe Daniel in Goldstein, Monique Ryan in Kooyong and Kate Chaney in Curtin all fit that category. Kylea Tink, who won the division of North Sydney in 2022, was inadvertently knocked out of the race by the Australian Electoral Commission, which [abolished](https://www.news.com.au/national/federal-election/electorates/independent-mp-kylea-tink-to-step-down-at-the-2025-election-after-aec-abolished-her-seat-of-north-sydney/news-story/291b7f60c35ac7934d9983eb8ad5c67f) her seat last year.

Andrew Gee, Russell Broadbent and Ian Goodenough are all incumbent MPs running as independents in seats where they were previously elected as Coalition candidates. Tasmania’s Andrew Wilkie, a long-serving independent with first-hand experience of a federal hung parliament, is seeking his sixth successive victory.

Bob Katter and the Centre Alliance’s Rebekah Sharkie also seeking re-election to the lower house, while in the Senate, crossbenchers such as David Pocock and Jacqui Lambie are all looking to retain their places. So is Coalition defector Gerard Rennick, who quit the Liberal National Party in Queensland over a [preselection loss](https://www.news.com.au/national/politics/lnp-senator-gerard-rennick-quits-party-to-contest-federal-election-as-independent/news-story/ef6cd6ec08b061d5edb906e4a9938704).

Rennick’s is perhaps the tallest order of that bunch, but none of them can take anything for granted. Even Katter, with his half-century of parliamentary experience and huge local popularity, is almost 80 and is facing a large field of younger challengers, all of whom will appear [above him](https://www.northwestweekly.com.au/eight-in-kennedy-race-as-lnp-draws-pole-position-2025-04-16) on the ballot paper.

**Campaign blues?**

Plenty of people have been watching national opinion polls during this campaign. But the polls are not terribly insightful for seat-by-seat contests involving large numbers of independent contenders. Even experienced pollsters [are saying](https://www.abc.net.au/news/2025-04-21/how-pollsters-pick-federal-election-winners/105193814) it has “never been harder to get pre-election polling right”.

Months out from the election, [polls conducted](https://www.thesaturdaypaper.com.au/news/politics/2025/03/22/polling-shows-teals-support-growing-coalition-base) on behalf of Climate 200 were showing possible wins for Heise in Cowper and Boele in Bradfield. Both could win. Heise has reportedly amassed a [formidable team](https://www.afr.com/politics/federal/teal-confidence-growing-as-liberal-support-slides-20250421-p5lt4a) of 3,500 volunteers in support of her grassroots campaign.

But the pressure and scrutiny of an election campaign can quickly put frontrunners under pressure. This is certainly true of Boele, whose campaign momentum stalled with a surprising scandal involving an inappropriate comment in a [hair salon](https://www.abc.net.au/news/2025-04-01/teal-independent-bradfield-nicolette-boele-apologises-hair-salon/105124018), as well as distancing herself from [allegedly antisemitic posts](https://www.smh.com.au/politics/federal/north-shore-teal-distances-herself-from-antisemitic-posts-20250421-p5lt2x.html) on her social media posts in 2022, saying a former volunteer was responsible for them.

Multi-cornered contests between defector MPs, the major parties and community independents will also make for interesting viewing on election night. Broadbent and Goodenough both seemed [quietly confident](https://www.afr.com/politics/federal/liberal-defectors-could-thwart-dutton-s-bid-for-power-20250421-p5lt4v) about their prospects when asked by the Australian Financial Review last week. The same cannot be said for Calare’s Andrew Gee, who began the election with a “[Facebook fail](https://www.theguardian.com/australia-news/2025/mar/29/mp-andrew-gee-praises-himself-as-the-good-guy-in-facebook-fail-ntwnfb)” and has since endured a [stressful](https://www.crikey.com.au/2025/04/17/2025-election-well-hung-calare-independents-coalition/) few weeks of bitter campaigning.

When it comes to winning back the seats that independents won last time, Liberal feelings range from [bullishness](https://www.theaustralian.com.au/nation/politics/election-2025-liberal-candidate-ro-knox-way-ahead-of-teal-allegra-spender-in-wentworth-says-compass-poll/news-story/8e660e96ed1121b4574628d30c445e92) to [bluster](https://www.abc.net.au/news/2025-04-22/kooyong-amelia-hamer-accuses-monique-ryan-of-dirt-file/105201894). Daniel faces a well-resourced campaign from her predecessor Tim Wilson in Goldstein and nothing is being spared in the contest against Chaney in Curtin.

In Kooyong, Ryan’s campaign has been hampered by the occasional error, such as her husband’s removal of an [opponent’s corflutes](https://www.abc.net.au/news/2025-03-24/federal-mp-monique-ryan-and-husband-apologise-after-sign-removal/105088472) and an awkward [exchange](https://www.news.com.au/national/federal-election/not-a-beauty-contest-peter-dutton-declares-labors-lies-are-hurting-him/news-story/ea4f13af29d1ebfa07934f468ff76634) with Sky News reporter Laura Jayes. In an election dominated by the housing affordability crisis, voters are less likely to remember these moments than the revelations that Ryan’s Liberal opponent, Amelia Hamer, a self-identified renter, [happens to own](https://www.theage.com.au/politics/federal/amelia-hamer-pitched-herself-as-a-renter-she-owns-two-investment-properties-20250404-p5lp8f.html) two investment properties.

The biggest drama has been in the affluent Sydney seat of Wentworth, where Spender has weathered attacks about her [political donations disclosures](https://www.afr.com/rear-window/allegra-spender-s-money-making-from-anonymous-donors-20250422-p5ltc7) and approach to tackling [antisemitism](https://www.smh.com.au/politics/federal/handshake-strong-words-first-public-face-off-for-spender-and-knox-after-legal-threat-20250406-p5lpki.html).

An anonymous person circulated 47,000 [leaflets](https://www.abc.net.au/news/2025-04-22/spender-wentworth-pamphlets-antisemitism-hidden-campaign/105198582) through the electorate criticising Spender’s “weakness” on antisemitism, flagrantly breaching electoral laws that require campaign material to be authorised. The Australian Electoral Commission has [identified](https://www.canberratimes.com.au/story/8949784/aec-identify-person-behind-unauthorised-poll-pamphlets/?msg=login) the culprit (said to have “acted alone”), but has been [less forthcoming](https://www.aec.gov.au/media/2025/04-24a.htm) about whether it intends to litigate the issue after the election.

**Making minority work**

It seems premature to start talking, as some pollsters have, about a Labor majority after May 3. It remains entirely possible crossbenchers may hold the balance of power, and in doing so, exert significant influence on the next government.

In the third leaders’ debate, Prime Minister Anthony Albanese, normally pragmatic, refused to countenance sharing power with other parties or MPs. Opposition Leader Peter Dutton made the surprising [admission](https://www.abc.net.au/news/2025-04-22/federal-election-2025-dutton-albanese-campaign-trail/105198544) he would willingly make agreements with independent MPs in order to win.

He certainly wasn’t thinking of the “teals”, whom he so often [berates](https://www.theage.com.au/politics/federal/federal-election-2025-live-updates-treasurer-claims-second-debate-a-decisive-win-for-pm-labor-slams-dutton-s-comments-on-russia-indonesia-military-plans-20250416-p5lsa6.html?post=p58idj#p58idj) as “Greens in disguise”. But there are others with whom he could easily work. Katter, Spender and Le are among Dutton’s [preferred](https://www.theguardian.com/australia-news/2025/mar/30/australian-election-2025-hung-parliament-chances-what-happens) negotiating partners. Sharkie has already declared that in a hung parliament scenario, she would [call Dutton first](https://www.theaustralian.com.au/nation/politics/ill-call-peter-dutton-first-sa-crossbencher-rebekha-sharkie-boosts-coalitions-minority-hopes/news-story/fff22bb54a3df9c9f0872e786179f83c).

There is no rulebook for making a hung parliament work. In the past, new political configurations and coalitions have been born from hung parliaments, including the [forerunners](https://theconversation.com/as-the-liberal-national-coalition-marks-its-100th-birthday-will-it-survive-for-many-more-199511) of the Liberal-National coalition.

Agreements can be limited to assurances of support on budget bills and confidence motions, or more expansive undertakings including policy commitments and institutional reform. In the event of a parliamentary impasse, crossbenchers can withdraw their support and allow a new minority government to be formed. The Australia Institute’s Frank Yuan recently [pointed out](https://australiainstitute.org.au/post/election-entree-parliaments-changing-the-government/) seven changes of government have been triggered by the withdrawal of crossbench support. Indeed, during the second world war, two independent MPs effectively changed the government [mid-term](https://theconversation.com/the-forgettables-5-australian-prime-ministers-you-may-not-know-much-about-196360).

Much depends on the relationships forged at the start of a hung parliament. In his [memoir](https://www.mup.com.au/books/windsors-way-updated-edition-paperback-softback), former New England MP Tony Windsor recounts the seventeen days of negotiations that followed the 2010 election. One of the factors that led him, along with follow independent Rob Oakeshott, to support the Labor Party was the “professionalism” and “respect” its leaders showed them. Former Coalition leader Tony Abbott, by way of contrast, gave Windsor the impression he was unlikely to endure minority government long enough to honour any of his commitments.

An especially aspirational crossbencher may even take on the role of Speaker. Wilkie and Sharkie have been recently touted as [contenders](https://www.smh.com.au/politics/federal/crossbench-mps-in-talks-with-labor-coalition-over-speaker-s-job-20250415-p5lryt.html) for the role in a hung parliament scenario.

**Reform hangs in the balance**

Independents MPs would be likely to bring particular policy priorities to any minority government negotiation. Given the heated contests in independent electorates, [truth in political advertising laws](https://australiainstitute.org.au/post/truth-in-political-advertising-laws/) would probably be high on the agenda. Steggall has previously promoted reforms to [Stop the Lies](https://www.thesaturdaypaper.com.au/comment/topic/2025/02/13/truth-advertising-should-be-urgent-priority), but when the Albanese government chose not to progress its own version of this reform, independents [signalled](https://www.afr.com/policy/economy/teals-to-demand-truth-in-advertising-laws-in-hung-parliament-20250114-p5l45f) it would be high on their priority list in a hung parliament.

Crossbenchers – in both houses – might also treat recent changes to Australia’s electoral laws as a bargaining chip. Those changes, agreed between Labor and the Coalition in secret, promised to get big money out of politics by imposing donation and spending caps on everyone but with [special caveats](https://theconversation.com/sweeping-reform-of-the-electoral-laws-puts-democracy-at-risk-they-shouldnt-be-changed-on-a-whim-249144) for major parties. Haines has [declared](https://www.abc.net.au/news/2025-02-13/helen-haines-political-donations-reform-minority/104929724) these are “in her sights” if a hung parliament arises.

The menu of reform options gets wider from there. Spender has [called for](https://www.allegraspender.com.au/4821/economic_reform) labour market and tax reforms that may not be palatable to all of her peers.

In the Senate (where “every day is minority government”), Pocock has outlined his [firm demands](https://www.thesaturdaypaper.com.au/news/politics/2025/04/12/exclusive-david-pococks-demands-minority-government) for greater royalties from resources rents and reforms to negative gearing and capital gains tax concessions. Energy and climate policy, as well as support for rural Australia, would likely figure in a larger negotiation.

The crossbenchers would be hard-pressed to agree on everything, but there is strength and wisdom in numbers. Albanese and Dutton are both very experienced parliamentarians. Crossbenchers would likely need to put their heads together to exert maximum leverage.

If there is a hung parliament after May 3, history shows us it can be put to good use. The 43rd parliament, in which the Gillard government was in minority, was one of the most productive in recent history. It passed 561 bills including landmark measures such as the Clean Energy Future package and its centrepiece, a carbon price. It also passed needs-based funding for Australian schools, the National Disability Insurance Scheme and [plenty more](https://theconversation.com/independent-mps-are-elected-for-a-reason-hung-parliaments-may-be-precisely-what-voters-want-251900).

That seems a decent enough model for the next parliament to emulate. After all, as Harold Glowrey seemed to appreciate nearly a century ago, not everyone needs to be a cabinet minister to play their part in shaping the future.
**Post 2:**

> Federal election 2025: Why Peter Dutton, Anthony Albanese campaigns are the worst I’ve seen **This election is one of the worst I’ve seen. Here’s the one thing we can do to fix Australian politics**

**Ross Gittins, April 14, 2025**

How are you going with the election? Are you getting a lot out of the debate, seeing the big issues canvassed and making up your mind who’ll win your vote?

It’s not as if the choice isn’t clear: do you want to wait 15 months for a permanent tax cut of $5 a week, rising to $10 a week a year later, or would you be eligible for a $1200 once-only tax cut in July 2026, plus an immediate one-year cut of 25c a litre in the price of petrol?

If that’s not enough to seduce you, there’s more. Anthony Albanese will cut the price of draught beer by 5¢ a glass for two years or, for small businesses, Peter Dutton will make entertainment expenses tax-deductible (conditions apply).

But you may want to judge it on the character of the leaders. Again, the choice is clear: do you want the controlled, experienced hand of Albanese, who’ll never do anything rash, whose goals are modest and whose motto is “steady but slow”? Got a problem? He’ll think about it. If you want a prime minister who’s on everyone’s side, Albo’s your man.

Or do you want tough cop Dutton on the beat, always quick on the draw and ready to protect us from the threatening world we live in? He’s heard of a supermarket worker who’d had a machete held to her throat. That won’t happen to you on Dutto’s watch.

And it’s not as if the campaign so far hasn’t been action packed. We’ve had [Albanese falling off the platform](https://www.smh.com.au/link/follow-20170101-p5louu) at an election rally, then denying it. We’ve had Dutton joining a kids’ football game and [hitting a cameraman in the forehead](https://www.smh.com.au/link/follow-20170101-p5lpex).

What would a campaign be without seeing pollies in safety helmets and high-vis vests on TV every night? Or at a childcare centre, showing how human they are and what good fathers they must be whenever they can make it home?

What’s new this time is Greens leader Adam Bandt taking a big red toothbrush with him to TV interviews (must have some meaning I’m missing) or his colleague waving round a bleeding headless salmon in the Senate.

What’s that? You don’t think much of the election campaign? It’s been neither interesting nor edifying, and hasn’t got to grips with the big issues?

Well, I agree. I think both sides are treating us like mugs. Maybe like the mugs many of us have allowed ourselves to become.

In my 51 years as a journalist, this is the 20th federal election campaign I’ve observed at close quarters, and I’m convinced they’re getting worse: more contrived, manipulative, transactional and misleading, and less focused on the various serious problems facing us, which are far greater than they used to be, and now include America’s abdication from leadership of the free world.

In short, election campaigns have become dishonest, aimed at tricking us into voting for one side rather than the other, using trinkets to distract us from the bigger issues that neither side has thought much about nor has any great desire to tackle.

I know that’s easy to say for an oldie like me (77, since you were too polite to ask). “It was much better in my day.” But though things weren’t great in the old days – we’ve never been a paragon of Socratic debate – I think they’ve got worse over the years, and I’ll try to show how they’ve got worse and explain why.

But I must say this: even if things in Australia have got worse, they’re not as bad as they are in many other countries, [particularly the US](https://www.smh.com.au/link/follow-20170101-p5loy4). Nor are they ever likely to be.

Three things protect us from other countries’ decline. First, [compulsory voting](https://www.smh.com.au/link/follow-20170101-p5a79a), which forces everyone to register a choice and pay at least some degree of attention. Second, preferential voting ensures the person who wins is the one most of us prefer.

And third, an independent electoral commission which regularly increases the number of electorates and redraws boundaries to ensure there’s roughly the same number of voters in each, and these have boundaries that aren’t gerrymandered to give one side or the other a built-in advantage.

This is in [marked contrast to the US](https://www.smh.com.au/link/follow-20170101-p5kkf5), where each state government determines its own federal voting arrangements. Their gerrymandering ensures they have very few marginal electorates, whereas we have a lot. And we don’t have voting arrangements designed to disadvantage certain classes of voters, such as racial minorities.

So we shouldn’t complain too much. Even so, our election campaigns have changed over the years, and not for the better.

They’ve changed because [the voters have changed](https://www.smh.com.au/link/follow-20170101-p5la56) – Gen Z seems a lot less interested in conventional politics than we Baby Boomers were at their age, when there was so much disapproval of Australia’s part in the Vietnam War, and so many young men (including me) hoping not to be conscripted.

Another important source of change is technological advance, particularly the effect of the information revolution, which has armed the parties with greater knowledge of voters’ views, and changed the media by which politicians reach out to voters.

Finally, the political class’s changing aspirations have affected the way campaigns are run. In the olden days – even before my time – politicians used to travel round, visiting key electorates and talking to voters. They’d do this at evening public meetings or, during the day, from the back of a truck in the main street.

But the advent of television changed all that. While local politicians and their supporters may canvas their electorates door to door, most contact between the party leaders and the voters occurs via TV.

These days, leaders still visit marginal seats around the country, but what they do during the day is aimed at producing the colour and movement that will get them a spot on the evening TV news – hence helmets and high-vis.

They’ve worked up a list of promises to announce, and they (and their media entourage) go somewhere vaguely relevant to deliver the announcement. Guess where they go to announce a change in childcare?

A big advantage of this is that their busy day ends late afternoon, once the TV news camerapeople have got what they need for this evening. Then the leaders can go to a fundraiser, appear on a current affairs program, or get an early night.

Trouble is, though the pollies haven’t changed their routine, the evening TV news bulletin isn’t nearly as universal as it was. When there were only four channels, all airing their news at the same time, if you wanted to watch telly while you had dinner, you couldn’t avoid the nightly news bulletin.

Now the proliferation of TV choices makes it much easier to avoid the news, which many do. The parties have started using social media to spread their messages, but this makes it harder for the rest of us to see what they’re up to.

As the proportion of people who don’t follow the news – and aren’t much interested in politics – has grown, the parties have had to reach them via advertising. They now spend a fortune on TV ads, with far fewer ads in print and on radio. My theory is that, for the many people who don’t follow politics but know they’ll have to vote, they do their last-minute homework by remembering the TV ads they’ve seen.

But advertising works by appealing to our emotions, not our brains. Don’t explain the details, just make me feel nice – or angry. The parties know negative ads – attacks on their opponents – work better than positive ones (“you’re gonna love my policies”), which is hardly a boon to the democratic process.

This is what has made fear campaigns – misleading people about how badly they’d be affected by the other side’s planned changes – so fearfully effective. And the increasing resort to fearmongering is a major way by which election campaigns have become less informative and more misleading.

So it’s not just the way the mechanics of campaigning have changed. More importantly, it’s the way what’s said has changed, and the way the politicians’ objectives and behaviour have changed.

Politics has become more professional. In former times, politicians tended to be men (yes, almost all of them men) who turned to politics after a career as a lawyer, businessman or union official. They’d wearied of making money and decided to spend the last part of their working life fighting for a cause.

I’m sure personal ambition has always been a big motivation for getting into politics, but in those days, it came mixed with a strong desire to make the world a better place. These days, politics has become a career path you follow for most of your working life.

When young people are interested in politics and would like to make a career of it, they get started as soon as they leave university, taking a job working for a union, or in a minister or opposition minister’s office. The number of people working in ministers’ offices has grown considerably during my time in journalism.

It started in the Labor Party, but then the Liberals joined in. You work your way up the ladder, first aiming for preselection as a parliamentary candidate. Once you’ve made it into parliament, you work towards a job as a minister or shadow minister, then see how far you can make it towards the very top.

Such a career path teaches you a lot about how the political game is played, but not much about how government policies work best in the interests of the public. It tends to replace any initial idealism with pragmatism and cynicism. It tends to feed ambition.

These days it’s rare for politicians to enter politics later in life. Two exceptions were the Liberals’ Dr John Hewson and Malcolm Turnbull. Both were hugely intelligent, and both cared about good policy, but both had trouble playing the political game at the professional level and didn’t survive at the top. Labor’s exception was Bob Hawke. His great success at the top came from all the politics he’d learnt while rising to the top of the union movement.

So when the barroom experts assert that most politicians care more about their personal advancement than about doing good things for the nation, I’m inclined to agree. As someone famous once said, “by their fruits ye shall know them”.

The professionalisation of politics is a main reason that what’s said and done in election campaigns has changed, but another reason is that politics has become more scientific. In former days, politics was played by ear. Pollies decided what voters liked and didn’t like from what the voters they met said to them, then used their own intuition to fill in the gaps.

These days, the parties spend a lot of money conducting private polling, not just of how people intend to vote, but what issues are more important to them at present. They also use carefully selected focus groups to get ordinary voters expressing their views on particular issues.

When someone says something and everyone round the table says “yes, that’s right”, the professionals running the group take note and pass it on to the pollies for them to use. Or it can work the other way: the pollies and their people think of lines to help sell a policy measure, and they’re tried out on appropriately chosen focus groups. What goes over well gets used in public utterances.

Between the careerism and the carefully gathered knowledge of what voters think, election campaigns have become more contrived. We’re transported to a fantasy land, where everything is nice and nothing is nasty (except the bad guys on the other side).

The pollies never try to tell a voter something they don’t want to hear. They never tell a voter they’re wrong about anything, and seem to go along with anything you may say, no matter how silly.

Have you noticed the way politicians expect us to be – and encourage us to be – completely selfish? It simplifies their job. They tell us what they can do for us and our families, never what we should be agreeing to in the interests of the country.

And the more they talk about doing this little thing or that little thing for us – the more they make following elections good preparation for a trivia quiz – the more they avoid having to talk about a host of big but controversial issues: climate change, the environment versus jobs, AUKUS, school funding, online gambling and even uninsurable homes. Of course, I couldn’t swear the media had played no part in dumbing-down election campaigns.

The pollies always tell us about the various nice things they plan to do to make our lives better, and never tell us of the not-nice things they’ll have to do to improve our lives. In election campaigns, every player wins a prize.

Not so long ago, a big part of elections was pollies being pressed to tell us exactly how much their promises would cost and exactly how they’d be paid for.

But doing that is what caused Labor’s Bill Shorten to lose the 2019 election he was expected to win. He had some expensive promises, but spelt out some small tax changes that would cover their cost.

These changes had been carefully selected to hit only some well-off people who could afford the loss, but the Libs ran a scare campaign telling ordinary punters they’d be hit and, with the effect magnified by a lot of yellow and black ads paid for by some fat Queenslander, Shorten lost enough votes to cost him the election.

The trouble here is that politicians on both sides have broken so many promises and said and done so many tricky things for so long that many voters have concluded they are all liars. This is why so many people have stopped listening to them.

But there’s one exception. The only thing a politician says that the doubters are prepared to believe is that their opponents are not to be trusted because they’re out to get you. “Ah yes, ain’t that the truth.”

That’s why scare campaigns have become the currency of election campaigning, with stultifying effect.

And that’s why the 2019 election has made elections and their campaigns much worse than they were. Under Albanese, Labor vowed never to be caught like that again. He made himself a “small target” at the 2022 election, promising to do very little, and not to do many things: introduce new taxes, increase existing taxes, and cancel or change the already legislated stage 3 tax cuts.

Apart from the latter, he’s kept those promises. He’s been a small-target prime minister, doing as little as possible to tackle our many problems, which is why so many of us are so uninspired by his performance. It’s far too risk-averse.

A leader who’s not game to do anything unpopular – such as putting up taxes – is a leader who’ll never make much progress solving our deeper problems, like giving our youngsters a fair shake, and never improve the future they profess to care about so much.

Trouble is, under our two-party system, when one side takes a position, the other side almost always copies it. We get less choice, not more. So when Albanese decides it’s safest to stay small target, Dutton stays small target.

When ditto Dutto keeps changing his policies mid-campaign, we’re watching him learning on the job not to be daring, not to fix things and, above all, to be only superficially different from the other side.

The big change since the 2019 election is that neither side will ever have the courage to propose any kind of tax change that would have some people paying a bit more – even those who could easily afford it. The tiniest possibility of an increase for some, and the fearmongers on the other side will soon have taxpayers throughout the land shaking in their boots.

This has taken the election-campaign fantasy land to a whole new level of unreality. The laws of economics have been suspended for the duration of the campaign. Government spending can only ever go up, while taxation can only ever go down. The budget deficit is presumed to be unaffected, covered by a sign saying Don’t You Worry About That.

Surely you remember the days when campaigns devoted much attention to “what do your promises cost and how will you pay for them?” That’s what tripped up Labor in 2019 and, I confidently predict, Dutton won’t let trip him up now.

Some worthy souls in the media keep lists of what the promises have cost and demand a detailed account of how that cost will be covered, but the two sides just brush them aside. They’re in tacit agreement not play that game any more. In truth, both sides will add to deficit and debt.

The other way to look at all this is that, by their poor behaviour – government by scare campaign – the two sides of politics have fought to a standstill. Neither side is game to do anything about any of our big problems for fear of the lies this would allow the other side to say about them. Now, I know what you’re thinking: “OK, Ross, if you’re so smart, what’s the solution to the mess election campaigns have got into?”

The good news is, the nation’s voters are already working on the solution. So many people have lost faith in the two sides of politics that the proportion of people voting for the two majors is the lowest it’s ever been.

In the 2022 election, the share of first-preference votes going to the minor parties and independents rose to almost a third, with the remaining two-thirds shared roughly equally between Labor and the Coalition. We saw the Libs losing seats to the teal independents, and the Greens winning more seats in the lower house.

I’m confident the minor parties’ share of the vote will go higher in this election. The experts are pretty sure that, whichever major gets more seats, it will be in minority government, needing the support of enough minors and independents to convince the governor-general it could govern effectively.

Both major parties would like us to believe minority government would mean chaos and no agreement on anything. Don’t be fooled. As we saw with Julia Gillard’s minority Labor government in 2010, the government was stable and passed more legislation than usual.

What changed was that, to get that support and stability, Labor had to agree to put through controversial measures it wouldn’t have been game to propose by itself. Such as? The carbon tax. Minority government transfers some power to the parties and independents who still believe we need real, controversial policy changes to solve our problems and improve our future.

So if you don’t like what the two major parties have done to campaigns and timidity in government, you should share my hope that this election puts neither major party back in majority government.
**Post 3:**

> The heavy price of keeping the west happy Western Australia made Anthony Albanese prime minister in 2022 and will vote to keep him in office only if the Albanese government honours what economist Saul Eslake calls “the worst Australian public policy decision of the 21st century” – and predictably Labor is honouring the deal.

The politics and economics of the west as they shape the Federation get virtually no attention in the east – yet they are critical to the nation’s unity and social compact and, in recent times, pivotal to who becomes prime minister.

There is one certainty. Within the Federation the power of the west is rising. The rise has been basic to the fate of both Scott Morrison and Albanese and who governs in Canberra. During the pandemic WA premier Mark McGowan ran the politics of separation to keep his state largely Covid free and became the most popular leader in Australian history. Prime ministers were hostage to him.

Leaders change but the power realities stay the same. Last December Treasurer Jim Chalmers announced that national cabinet – the Prime Minister and premiers – had agreed on more support for the struggling National Disability Insurance Scheme with the trade-off that the federal government would continue to honour the GST distribution deal negotiated in 2018 by Morrison when he was treasurer.

That deal is primarily about the west. It is probably unprecedented in Australian history. It is driven by political power, the power of a GST deal to shape the results of WA seats in federal elections. The deal negotiated by Morrison was bipartisan; under Chalmers it remains bipartisan. It is a special arrangement that delivers huge financial support to the richest state. Neither Labor nor the Coalition like to discuss what’s happening.

That riles independent economists Eslake and Chris Richardson, who have separately reached the same conclusion and campaign against what they condemn as a disastrous public policy with no redeeming features. “This represents a transfer of almost $40bn to the government of the richest state in Australia,” Eslake tells Inquirer. “A state that is richer than the rest of Australia by a vastly larger margin than any other state has ever been, so the citizens of that state can enjoy better public services and lower levels of state taxation than the citizens of the rest of Australia.”

Richardson tells Inquirer: “This deal is not about protecting widows and orphans. It is about funnelling money to the best-off state in the nation. If you are worried about Australia’s social compact, whether we have got this right, and if you’re looking for the dumbest thing we do, this is the stand-out.”

Saturday, February 17, 2024
Hi

HOMEINQUIRER
PAUL KELLY
The heavy price of keeping the west happy
Anthony Albanese, Scott Morrison and Malcolm Turnbull all reaped political rewards from the GST deal with Western Australia.
Anthony Albanese, Scott Morrison and Malcolm Turnbull all reaped political rewards from the GST deal with Western Australia.
12:00AM FEBRUARY 17, 2024502
Facebook
Twitter
Whatsapp
Email
Save
Western Australia made Anthony Albanese prime minister in 2022 and will vote to keep him in office only if the Albanese government honours what economist Saul Eslake calls “the worst Australian public policy decision of the 21st century” – and predictably Labor is honouring the deal.

The politics and economics of the west as they shape the Federation get virtually no attention in the east – yet they are critical to the nation’s unity and social compact and, in recent times, pivotal to who becomes prime minister.

There is one certainty. Within the Federation the power of the west is rising. The rise has been basic to the fate of both Scott Morrison and Albanese and who governs in Canberra. During the pandemic WA premier Mark McGowan ran the politics of separation to keep his state largely Covid free and became the most popular leader in Australian history. Prime ministers were hostage to him.

During the pandemic, prime ministers were hostage to WA premier Mark McGowan. Picture: Sam Ruttyn
During the pandemic, prime ministers were hostage to WA premier Mark McGowan. Picture: Sam Ruttyn
Leaders change but the power realities stay the same. Last December Treasurer Jim Chalmers announced that national cabinet – the Prime Minister and premiers – had agreed on more support for the struggling National Disability Insurance Scheme with the trade-off that the federal government would continue to honour the GST distribution deal negotiated in 2018 by Morrison when he was treasurer.


That deal is primarily about the west. It is probably unprecedented in Australian history. It is driven by political power, the power of a GST deal to shape the results of WA seats in federal elections. The deal negotiated by Morrison was bipartisan; under Chalmers it remains bipartisan. It is a special arrangement that delivers huge financial support to the richest state. Neither Labor nor the Coalition like to discuss what’s happening.

That riles independent economists Eslake and Chris Richardson, who have separately reached the same conclusion and campaign against what they condemn as a disastrous public policy with no redeeming features. “This represents a transfer of almost $40bn to the government of the richest state in Australia,” Eslake tells Inquirer. “A state that is richer than the rest of Australia by a vastly larger margin than any other state has ever been, so the citizens of that state can enjoy better public services and lower levels of state taxation than the citizens of the rest of Australia.”

Richardson tells Inquirer: “This deal is not about protecting widows and orphans. It is about funnelling money to the best-off state in the nation. If you are worried about Australia’s social compact, whether we have got this right, and if you’re looking for the dumbest thing we do, this is the stand-out.”

Making clear that the west will not tolerate any change, state Treasurer Rita Saffioti says Albanese Labor will risk losing “every seat in WA” if it reneges on the GST deal. Not a bad threat as threats go. But it’s not necessary, since Albanese and Chalmers are on board.

Saffioti’s threat, however, is beneficial. It confirms what the entire saga is about – West Australian seats. Richardson says of the threat: “It goes to the raw politics. We’re robbing the nation blind and you can’t stop us now.” Being fair to Chalmers, he pledges a Productivity Commission inquiry into the GST arrangements in 2026, yet Morrison’s initial decision drew on a PC inquiry, admittedly with loaded terms of reference.

Quizzed about the initial GST deal, Malcolm Turnbull justifies the decision: “There are a lot of things that work in theory from an economic point of view, but they don’t fly in political terms.” How important was the deal to Turnbull’s future given his narrow 2016 majority? “We’d have lost almost every seat in that state (WA) without it,” he says. In short, the GST deal was a survival necessity.

But it was Morrison, as prime minister in 2019, who became the beneficiary of the deal he had done as treasurer. In Morrison’s 2019 election win, he retained all the government’s seats in WA. In the west, the politics of the GST distribution extend statewide and have a statewide impact.

Turnbull recalls that when doing the deal Morrison had a row with then Tasmanian treasurer Peter Gutwein. A concerned Turnbull asked Morrison: did you use the F-word? Morrison conceded he had used the M-word. What’s this more terrible M-word, Turnbull asked. “Mendicant,” said Morrison. All was revealed.

Morrison rejects the critique of the deal made by the economists – on both financial and political grounds. “This guarantee was never a permanent arrangement,” he tells Inquirer. “It was a time-limited arrangement. It was a transition. Under the old system some states were receiving gains from a model that was out of date with Australia’s modern economy.”

The origins of the west’s deal lie deep in our history. Hardly any schoolchildren in the east know that in 1933 the people of the west voted by referendum to secede from the federation, a step thwarted by Britain. Few people grasp that we still live with the consequences.

The government of Joe Lyons authorised the Commonwealth Grants Commission to provide “impartial and independent” advice on the distribution of grants to the states, the upshot being the commission’s 1936 report that grants reflect an equity principle to enable states to deliver similar levels of government services.

When the GST was created these principles were refined as the basis of distribution to the states. For much of the 20th century WA, given its small population over a vast area, was a beneficiary getting extra funds along, of course with Tasmania. But the west was transformed by the various stages of the mining and resources boom.

Eslake adopts an infamous Paul Keating line, saying the west got “kissed on the arse by a rainbow”. Of course, it’s unfair, given the WA resources sector is Australia’s most globally successful industry. But it captures the scale of transformation; the west, long one of the weakest financial states, is now the financial powerhouse of the Federation. Its scale is unrecognised.

That may be a good or bad thing. Eslake’s analysis is that the west’s per capita gross state product is running at more than 60 per cent above the national average over the past three years – that’s up from a modest 7 per cent above the national average at the turn of the century.

He tells Inquirer: “To the best of my knowledge there has never been another time, at least not since Federation, when one state has been richer – as measured by per capital gross product – than the rest of the country by as large a margin as Western Australia has been in recent years.”

By way of comparison, Eslake says during the past 45 years the ACT’s per capita gross product has never been more than 24 per cent above the national average; history suggests Victoria’s has never been more than 12 per cent above and NSW never more than 8 per cent above the national average.

A student of Federation and a loyal Tasmanian, Eslake – who returned after three decades on the mainland to live outside Hobart in his beloved island state – has despatched his analysis to the federal Treasury. It shows, among other things, the value of the west’s iron ore production has risen from just $3.7bn in 1999-2000 to an average of $111bn a year in the past six years.

He says the iron ore price rose to over $US100 a tonne by the end of 2009, stayed above that level until mid-2014, fell to a low of just over $US40 a tonne at the end of 2015, but then rose again and has been above $US100 a tonne for most of the time since June 2020. In addition, there has been a vast increase in the volume of WA’s mineral and energy production since 2010.

Given its financial strength, WA’s share of the GST funds then fell sharply. The west highlighted its complaint – saying it was getting “less than 30c in the dollar” of GST revenues it contributed. The lament of unfairness and being “ripped off” by the east was ritualised by the local media. Morrison, a highly political treasurer, responded. Morrison’s deal put a floor under the relativity of each state to guarantee its proportion of GST revenue would not be less than 70 per cent, and then 75 per cent, of what it would receive under an “equal per capita” distribution. The floor, of course, was only relevant for the west. It was a deal for WA.

But if WA got more, it meant the other states got less – a disaster. So Morrison agreed to “top up” the GST pool so that, until 2026-27, no state or territory would be worse off. In short, WA was the big winner, there were no losers and the federal government – or the Australian people – picked up the extra bill. The national interest was the loser.

It was initially estimated to cost the federal budget $8.2bn across eight years; Eslake now estimates it has blown out to $39.2bn across 11 years and “could end up being as much as $50bn”. That’s because the higher commodity prices, the richer is WA, the more the commonwealth must contribute to honour the 70 and 75 per cent in the dollar WA pledge.

As prime ministers, Turnbull, Morrison and Albanese have all signed up. The motive in each case has been the same – seeking to hold or win WA federal seats.

In truth, the politics of the GST distribution is diabolical. That’s because the model – based on giving and taking – institutionalises winners and losers by its design.

That’s why Morrison adopted the “no worse off” extra funding provision. It was a political necessity. This harks back to one of Turnbull’s replies to his critics: the job of government is not to follow economic purism but to find solutions that work for governments and people.

Morrison tells Inquirer: “The way the GST distribution had been working was indefensible. Western Australia was put in a ridiculous position. The idea that you could have a state getting less than 70c in the dollar from the GST distribution was unfair. That was undeniable.” This is Morrison speaking as a political realist. He says there are limits to how far a state can be penalised because of its financial strength. He says he sought not just a better deal for the West but a better distribution system overall.

However, Eslake and Richardson have had enough. Lacking any political power, they use the weight of argument to highlight the financial and equity anomalies embedded in this bipartisan compact with the west.

“We’re wasting a lot of money on the NDIS but at least it’s a noble cause,” Richardson says. “This is an enormous amount of money being spent, not on a noble cause but on a political stitch-up. It’s a backroom deal. This is the biggest amount of money we are spending on what achieves the least for Australia.”

Turnbull says: “Delivering durable GST reform was, and will remain, a long-term asset for the Coalition in WA.” Well, that depends. Now, it is an indispensable asset for the Albanese government in WA but there is no certainty it will suffice for Albanese in the west at the next election.

Eslake says the initial cost of the “no worse off” guarantee was predicated on the assumption that iron ore prices would fall back to and remain at $US55 a tonne – meaning that in turn the guarantee wouldn’t cost the budget too much. He says that with the federal budget now in structural deficit – despite its current, transitionary surplus – the federal government presides over a transfer of almost $40bn to the WA government, “the only government in Australia, and one of the few in the world, which is running and expects for the foreseeable future to run persistent budget surpluses – so that they can run even bigger ones”.

He says: “How that can be reconciled with any serious concept of equity, let alone fiscal prudence, is surely beyond comprehension.”

Eslake says the distribution of revenue from the GST is the “largest single expenditure program in the federal budget”. It totals $92.5bn in the 2013-24 budget, far above the cost of the Age Pension or the NDIS. He accepts the principle at work – the stronger states support the weaker states in the cause of delivering the goal of government services equality. His objection is to the WA redistrib­ution

Asked about the scale of the policy failure, he says: “I would say it is a corruption, not in the NSW police force sense but a corruption relating to the biggest spending program in the federal budget.

“The principle here, that it shouldn’t matter where you live to determine your access to health and education, that principle goes back to the 1930s. That’s when it was first promulgated by the Grants Commission in the aftermath of the referendum in Western Australia. It should be upheld.”

The story of the impact of the west on national politics is heavily tied to Morrison. One of his most disastrous decisions – perhaps the most fatal he made as prime minister – was initially to back Clive Palmer’s High Court challenge to the pandemic-induced border closure by McGowan. McGowan tried to dissuade him, but only too late did Morrison realise the damage and withdraw from the challenge. But he had given Albanese his campaign: that Morrison wanted to sabotage the closed border that kept the west safe from Covid.

Albanese is Prime Minister today courtesy of the west, where he won four seats to deliver his one seat 2022 election victory. Morrison’s success in the west at the 2019 election, when he held 11 of the 16 seats, turned into ashes at the 2022 election.

WA is now the strongest fiscal state in the nation. Indeed, most West Australians are proud of this fact. They have every reason to be proud. It is also a fact WA governments, McGowan in the lead, have extracted huge benefits for their state. That, after all, is the job of a premier. McGowan, in retrospect, shouldn’t be criticised. His achievements were extraordinary.

Asked why he was campaigning to unwind the GST deal when both major parties supported it, Eslake says: “Until three weeks ago both major parties supported the stage three tax cuts and now neither of them does.

“The Prime Minister’s defence against the charge he had broken a promise – which he has done – is that circumstances around the stage three tax cuts had changed dramatically since they were legislated. But haven’t the circumstances around this issue also changed dramatically?” Eslake points to the cost explosion in the policy and the fact the federal budget, expected back in 2019 to be heading into surplus, is now projected to be in a long-run structural deficit. They constitute a significant change in circumstances.
**Post 4:**

> Flawed cashless welfare cards rebadged # Flawed cashless welfare cards rebadged

April 19, 2025

Minister for Social Services Amanda Rishworth. Credit: AAP Image / Aaron Bunch 

*Despite promises to end the Coalition’s Cashless Debit Card, Labor has rebranded the welfare payment system that is compulsory in some Indigenous communities.*

By Rick Morton.

A full parliamentary term after promising to end income control, the “suffocating” and “humiliating” policy continues for almost 30,000 people – despite being overwhelmingly rejected in unpublished submissions to the latest consultation over the future of the scheme.

Although the Albanese government began the process of ending the Coalition’s Cashless Debit Card (CDC) early in its term, briefing notes sent within Services Australia in October 2022 requested a $21.5 million tender for the card’s provider, Indue, to “support participants to achieve a minimally disruptive transition to income management”.

Essentially, it was a tender to allow Indue to continue operating a rebadged, compulsory income management program.   

“The agency intends on leveraging the existing CDC technology enabling participants to continue using their cards,” the tender said, “but under a different product name and contract.”

The program continues to grow under Labor, and the Coalition has vowed to bring back the CDC “in communities that want it”.

“They want that card back,” the shadow minister for child protection and Indigenous health services, Kerrynne Liddle, told the ABC in January. “They see a direct correlation, and have experienced the direct correlation, between the card’s removal and what’s happened to them now.”

For political reasons, both the Coalition and Labor speak as if the end of the cashless debit card also spelt the end of income control. The opposite is true.

Under the renamed system that replaced the CDC, known as Enhanced Income Management, there are now 20,007 participants, 79 per cent of whom are Indigenous and all but 4 per cent of whom were forced into the scheme without any say.

In addition to these, a further 11,867 people – 87 per cent of whom are Indigenous – are still on the original version of income management that has been around since the Howard government’s Northern Territory Intervention in 2007.

This system uses an old model BasicsCard that requires a PIN and does not attach to a regular bank account. The CDC and its replacement, the “enhanced” income management, use newer technology that functions like a regular bank card.

Labor has called its version the SmartCard but, like all three iterations, it quarantines between 50 and 90 per cent of welfare funds and is designed to block purchases of products such as alcohol, tobacco, pornography and gift cards or items that can be easily sold for cash, as well as preventing cash withdrawals or spending on gambling.

In establishing new arrangements, Social Services Minister Amanda Rishworth introduced two new sets of legislation and corresponding legislative instruments that go further than what the Coalition was able to achieve in its aborted attempt to roll out the CDC universally in the Northern Territory.

These new powers allow any minister to extend income management to any new location without legislation. The Parliamentary Joint Committee on Human Rights said in 2023 that “the bill and related instruments extend all measures relating to income management to the enhanced income management regime … in effect, the legislation remakes the law relating to income management and possibly expands its scope”.

“People already vulnerable are further exploited as they sell what’s on their card for a lesser cash amount. Those who have previously had financial abuse are subject to further abuse. Money on the card can only be spent in large stores.”

Uniting Communities chief executive and Accountable Income Management Network convenor Simon Schrapel told *The Saturday Paper* the Labor government moved quickly to terminate the CDC when it won the last election but has since expanded the underlying scheme of income control.

“It was a great disappointment, really, because we engaged with the government in those early days and they acted quickly with the legislation to end the Cashless Debit Card and then they put this thing in called Enhanced Income Management, which was really a bit of sleight of hand,” he says.

“We’ve all been duped and we are deeply disappointed. The consultations that have been done have just stalled the process and we’re not entirely sure what is motivating that, whether it’s the bureaucracy that has an issue about wanting to keep this in place or whether there are particular government ministers that are still committed to some form of income management.”

Last year, the parliamentary human rights committee, chaired by Labor MP Josh Burns, recommended social security legislation be amended to explicitly make income management voluntary. This has not happened.

Instead, the Labor government promised yet more consultation into the future of the various schemes. The latest round ended in early December but, unlike other public consultation processes, the Department of Social Services has chosen not to publish submissions received on its website, despite gaining permission from people to do so.

These submissions were eventually disclosed through an order for the production of documents in the Senate and provide insight into what the government has heard about the scheme.

“A flawed, cruel and expensive set of restrictions on people’s economic independence that should never have been drafted, never mind implemented,” one person wrote. “Income management \[IM\] isn’t necessary except in extreme individual circumstances and should never be applied as a blanket measure. This policy has led to evictions due to recipients being unable to reliably pay rent via their income managed card. It has led to people being unable to buy essentials in power or tech failures. It prevents people from participating in legal activities where cash is the only payment method as 20 per cent of an income support payment is very little money to ‘spend freely’.

“I could go on but please, this policy is a punishment directed at vulnerable people who are, by necessity, excellent at balancing a limited budget.”

The cards do not work the way government claims they do. The product-blocking technology that is supposed to identify “forbidden” items at the point of sale is notoriously patchy and the new SmartCards that allow the convenience of tap-and-go payments for individuals are easily exploited.

For those who want to find a way to liquidate their quarantined funds, they do so at a loss.

“I work in youth homelessness services, IM doesn’t work,” one person told the consultation. “People already vulnerable are further exploited as they sell what’s on their card for a lesser cash amount. Those who have previously had financial abuse are subject to further abuse. Money on the card can only be spent in large stores.”

National Regional, Rural, Remote and Very Remote Community Legal Network (4Rs Network) co-convenor Judy Harrison tells *The Saturday Paper* the current system of compulsory income management captures most people based on geographical location, not whether they actually “need” income management.

“So the only way that tens of thousands of people, or any large number, can be warehoused like this on compulsory income management is by mistreating them,” she says.

“There aren’t the resources in the department to do an individual assessment. So that means we can’t have criteria that would require them to be individually assessed, with the onus on the department, because we can’t afford to administer that system.”

As it stands, people can apply to leave compulsory income management but the process is convoluted and the bar for acceptable evidence so high that instances of opt-outs are vanishingly rare.

Harrison said the adult guardianship and trustee system – which can see people with severe mental ill health or other incapacities have their personal or financial affairs managed on their behalf – is legislated and requires a rigorous and reviewable tribunal process before any serious decision like that is made.

“Now compare that with the cashless debit card where people are just put on it – they’re not put on it as individuals, they’re put on as a group and for the high majority it is done geographically,” she said.

“I just find it really remarkable that somehow, the scale of what’s involved in intruding on somebody’s finances hasn’t registered as being a moment, a major human rights and legal event, a major societal event when in other contexts we’ve got all these other checks and balances that don’t always work, but they’re there and we know they’re needed because every one of us, as an individual, has rights.”

Rishworth has requested or received multiple briefings from her department about the future of income management, most notably one summarising every media mention of the abolition of the CDC in 2023 and 2024 – a document that runs to 13 pages.

In another, the talking points anticipate Rishworth being asked about the government’s broken promise to end mandatory income control. The briefing anticipates two questions the minister might be asked on the topic: “Why hasn’t the Government ceased compulsory Income Management yet, as recommended by their own Senators in the Community Affairs References Committee report on the ‘Extent and nature of poverty in Australia’?

“Why do enhanced Income Management legislative instruments operate far beyond when the Government committed to abolishing compulsory Income Management?”

Answering its own question, the suggested response offered to the minister is: “Once consultation is complete and further decisions are made on what the future of the programs looks like, additional legislative changes will be made. This will include reviewing the ongoing requirement for these instruments.”

As a result of this indecision, Simon Schrapel says, the infrastructure for dramatic expansion of income management is in place for any future government.

“Clearly the opposition has a policy position of reinstating the cashless debit card and probably extending it much further in terms of its reach, so leaving the infrastructure and the technology in place makes it a whole lot easier,” he says. “So if there’s a change of government, I think it’s going to be a whole lot easier for an incoming government to ramp things up really rapidly.”

The irony is that Labor made cashless welfare a big feature of its election campaign in 2022 and helped fan the flames of a panic that the Coalition had already drawn up plans to apply income management to age and disability pensioners. This time around, there is little to say.

During a keynote speech at the McKell Institute in Sydney on Tuesday, Rishworth rattled off a roll call of achievements in her first term, including raising the base rate of working-age and student payments by $40 a fortnight but didn’t mention the cashless debit card or its replacement.

When she came to office, Rishworth said, “trust had been shattered between government and community by the robodebt scandal and income support recipients had been demonised”.

In December, the new conservative chief minister of the Northern Territory, Lia Finocchiaro, demanded the federal government “implement 100 per cent income management for parents of youth offenders” as part of her suggested plan to combat crime.

As the Coalition makes its intentions clear, Labor has failed to reaffirm its one-time rejection of compulsory income management.

“We’ve been trying to get a sense of, well, what’s next?” Schrapel says. “They know what the opposition have said and there is a chance for the government to actually differentiate. We do need to actually get an answer.

“Are they prepared to come out before May 3 and actually say, ‘We will, in the first 12 months of being re-elected, ensure that there is no form of compulsory income management in Australia again?’ Or will they do another three years of consultation? They won’t say what their plan actually is.”

A campaign spokesperson answered on behalf of Rishworth and Minister for Indigenous Australians Malarndirri McCarthy.

“The Albanese Labor Government committed at the last election to abolish the Cashless Debit Card and to make it voluntary in those communities through the SmartCard. We have delivered on this commitment,” the spokesperson said. “We’re delivering a long-term plan to reform income management, which has been in place since 2007, and are committed to working through this matter in partnership with the communities that would be affected by any changes.”

\*This article was first published in the print edition of The Saturday Paper on April 19, 2025 as "Cashless society".\*Flawed cashless welfare cards rebadged
**Post 5:**

> Exclusive: David Pocock’s demands of a minority government David Pocock wants a far greater slice of Australian gas export income through the Petroleum Resource Rent Tax, and the reform of capital gains and negative gearing tax breaks. These are the crossbencher’s two top demands for whichever party seeks to form government after the election, as part of his broader integrity agenda in the 48th parliament.

The independent ACT senator has cast off Climate 200 support in 2025 as he again vies with Labor’s Katy Gallagher, as well as a low-profile Liberal challenger who is seeking Canberra’s “contractor vote”. On this issue, Pocock is leaning confidently into the federal Coalition’s Trump-style attacks on the public service.

“Every day is a minority government in the Senate. I’ll work with whoever is in there, but I won’t tolerate the kind of Canberra-bashing we have seen and a plan that will decimate the Canberra economy, the ACT economy,” Pocock tells *The Saturday Paper*.

“The thing that people need to understand, and I think are starting to realise, that when you say, ‘We’re going to cut 41,000 public servants’ – even if not all of them are from Canberra, if a big chunk of them are from Canberra – that’s a huge impact on small businesses in the ACT.

“You can’t just remove public servants and not have an impact on other sectors of the ACT economy.”

[https://www.youtube.com/watch?v=5b9S2JL-XZY&ab\_channel=TheSaturdayPaper](https://www.youtube.com/watch?v=5b9S2JL-XZY&ab_channel=TheSaturdayPaper)

The former Wallabies captain is seeking a second term as an ACT representative after his election in 2022 – territory senators face voters every three years instead of the usual six. With current voting trends, neither Labor nor the Coalition is expected to secure a majority in the Senate at this election.

Both Anthony Albanese and Peter Dutton have ruled out cutting deals with the Greens should they need to form minority government.

Pocock sees the possibility of a hung parliament, predicted in most major opinion polls, as a way to deliver reform to address debt reduction, the environment and housing challenges.

He nominates the resource rent tax and housing tax reform as the “low-hanging fruit” of the next parliament.

“Both major parties jump up and down about budget deficits, structural reform and then do exactly nothing to actually change things when it comes to revenue and structural reform. Why would we give away half of our gas for free? Export LNG has brought in zero cents of Petroleum Resource Rent Tax. Ridiculous.”

To tackle the housing affordability crisis, Pocock wants housing treated as a human right and “more courage” from the major parties. He is not pursuing the sweeping agenda of the Greens, however.

“We have to look at the capital gains tax and negative gearing,” Pocock says. “I don’t think it’s a case of you either leave it as it is or you just scrap everything.”

With the Coalition having largely opposed key government legislation in the last parliament, Labor required support in the Senate from the Greens and crossbenchers such as Pocock, Jacqui Lambie and Lidia Thorpe. Key housing legislation was held up for months by the Greens, who are also eyeing the balance of power in a possible minority government. They want the capital gains tax discount and negative gearing scrapped but are offering exemptions for people with one investment property.

“Senator Jacqui Lambie and I had a range of measures costed,” Pocock says. “I think in that there’s some really sensible ways to turn it around, including by grandfathering existing arrangements. People have made investments based on the current rules. You may not like the rules, but they have been the rules.”

It will be no simple negotiation if Labor is on the other side of it. Labor took these two property tax reform proposals to the 2019 election – a platform that some blame for former party leader Bill Shorten’s defeat.

Albanese has repeatedly rejected any wind-back of tax breaks for investment properties, particularly in relation to housing policies from the Greens.

Out on the campaign trail, the prime minister was asked bluntly, “Can you rule out any changes to negative gearing and capital gains tax settings if re-elected?” Albanese responded tersely, “Yes. How hard is it? For the 50th time.”

Treasurer Jim Chalmers has also scoffed, saying, “We’ve got our own agenda on housing.”

Pocock also wants to steer the major parties onto matters of integrity. Like independent MP Helen Haines and the Greens, he says the National Anti-Corruption Commission ought to be subject to an expedited statutory review and to have far more open hearings. He also wants gambling reform pursued in the next term, in line with the wishes of late Labor MP Peta Murphy: a total ban on gambling ads.

“I’m constantly pushing senior public servants to do better,” Pocock says. “Yes, we should have high expectations. We should be spending money well. The way to do that is to actually fully fund something like the Australian National Audit Office, which the Coalition severely underfunded and Labor haven’t fully funded.

“They still can’t do as many audits as they’d like. That’s a real indictment on both of them. That should be your starting point. And then let’s start to look at things like procurement.”

Liberal sources tell *The Saturday Paper* that Pocock could have bargained harder with Labor in his first term to clinch concession, particularly for the ACT.

In the territory race, Pocock’s main rival is Labor’s first pick Katy Gallagher, a former ACT chief minister who in her subsequent federal ministerial career has become the most powerful politician Canberra has ever produced. She, too, is heavily focused on the Liberals’ attacks on the bureaucracy.

“Pocock has made it clear he’d work with anyone. That’s the position he’s taken as an independent,” the minister tells *The Saturday Paper*.

“A Liberal government would decimate this city regardless of whether Senator Pocock is on the cross bench. They’ve basically declared war on our town. They’ve disrespected us. They disrespect the work that we do, all the roles that we play in the nation.

“The only way to stop that is to stop \[Dutton\] being elected. And the only way to do that is to vote Labor. It’s pretty clear. That’s very clear in my head.”

Despite being seen as a Labor town with the party holding all three lower house seats and Gallagher’s Senate seat, there is a solid block of Liberal voters in Canberra, and she regards the three-way tussle with Pocock as making the ACT marginal and challenging.

In 2022, Gallagher’s campaign shifted to a defensive “Keep Katy” mode as it became apparent the Labor vote was under threat from either strategic voting or complacency from traditional voters.

Pocock ended up defeating conservative Liberal minister Zed Seselja for one of the ACT’s two Senate seats, but the numbers showed that while he peeled off disaffected Liberal voters, he was more successful in carving off progressive votes from Labor and the Greens.

Gallagher expects Pocock to beat her to fill the seat quota in his second-term quest.

“I do think Pocock is very popular, and I think there’s a level of complacency about support for me in the sense that a lot of people say, ‘Oh, Katy’s elected,’ ” she says, also referring to a “rusted on” Liberal vote in the ACT of about 25 per cent.

The Liberal Senate candidate Jacob Vadakkedathu – the owner of a small consultancy company – had to campaign in the context of his party’s policy to slash the public service by 41,000 positions. This is the total roles added since Labor took power and switched capacity away from expensive consultants. The Coalition’s stated focus on Canberra for the cuts would have meant laying waste to 60 per cent of bureaucrats based in the capital.  The backtrack announced this week by Peter Dutton means the proposed cuts would be achieved only by a hiring freeze and natural attrition.

At the same time, he also abandoned the policy to force public servants back to the office. Local Liberals say they had sway. “It’s a big win for us that we got the change. Don’t think these things happen without conversations,” a party source tells *The Saturday Paper*.

Requests from this paper for an interview with Vadakkedathu did not receive a response.

The Liberal candidate has been media shy throughout the campaign, but he gave an early interview to ABC Radio in which he backed Dutton’s planned cuts to the Canberra bureaucracy. He also defended the Liberal leader’s decision, should he become prime minister, to take up residence in Sydney’s Kirribilli House instead of the Lodge in Canberra, saying the comment was “taken out of context”.

The Liberals reject any notion they have given up in the ACT, saying they are running a “very traditional, finance-based, cost-of-living campaign for the average Canberra family”.

The party sees a significant opportunity in the number of Canberrans who have had to leave lucrative government contract positions and may want to blame Labor at the ballot box.

“There is a cohort of people who much prefer to work as contractors and their lives have been severely curtailed under Katy’s leadership in the Senate,” the Liberal source tells *The Saturday Paper*.

“We meet them over and over again. We have them in the party. We meet them on the hustings. They loved their life as contractors. It just doesn’t suit the Labor narrative, you see. They were paid more. They took more risk. Now some of them are employees of departments because they are still needed to do work. They would prefer to go back to being contractors if it was stable and reasonable.”

Gallagher says contractors were often asked to do roles of public servants, not the more specialised roles they wanted to do. “They’re consultants or contractors for a reason,” she says.

In his ACT campaign, Pocock says he will keep pressing for a “city deal” to attract more investment to the national capital – a Coalition-era initiative that Labor has not been keen to revive. Pocock has used his position to extract local benefits, such as helping to restore ACT access to assisted dying, an Upper Murrumbidgee River package and cancer support for ACT firefighters.

If Labor and the Liberals are serious about public service efficiency, says David Pocock, then there should be better funding for established independent mechanisms to improve it.

On the issue of campaign funding, the independent senator had a $1.79 million “war chest” in 2022 and just over $850,000 came from the political fundraising vehicle Climate 200. There was also a number of large donations from wealthy investors and many small individual donations.

Pocock says he’s passed up Climate 200 funding in 2025, and he’s never wanted to be seen as a “teal” independent in the Senate.

At the halfway point of the campaign, he is running on half the amount of donations that came in over the full 2022 race.

“I didn’t feel like I needed the money. I think they’re \[Climate 200\] really useful and important for new campaigns, but as an incumbent you have all the incumbency advantages,” the senator says. “You’ve got a team, you more or less know what you’re doing, and I want to stand on my record and back myself to be able to raise money based on what I’ve done.

“And I think we’ve seen that. People are keen to support community-backed independents that are in there fighting for them.

“We’ve had way more smaller donations. So, it will come out with declarations, and we can sit and compare and contrast.”

Asked by *The Saturday Paper* if the large individual donations of the last campaign are happening again, he said: “I think there’s been a few, not to the same scale as last time.

“This time, I’ve had over a thousand people contribute to my campaign. It’s far, far leaner.”

With minority government in the Senate set to be returned on May 3, there have so far been no overtures from either of the major parties. Pocock knows he is a competitor.

*This article was first published in the print edition of The Saturday Paper on April 12, 2025 as "Exclusive: David Pocock’s demands of a minority government".*
